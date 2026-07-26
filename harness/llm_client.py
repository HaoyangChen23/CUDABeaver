"""
LLM API client module.

Extracted from run_experiment.py for independent testing and swappability.
Supports two backends:
  1. OpenAI-compatible API (GPT, Qwen cloud, DeepSeek cloud, etc.)
  2. vLLM-served local models (via OpenAI-compatible endpoint)

Backend is auto-detected from config["api"]["backend"]:
  - "openai" (default): real OpenAI API or compatible cloud providers
  - "vllm":  locally served model via `vllm serve`

Key differences handled:
  - vLLM: api_key optional (defaults to "EMPTY"), no reasoning_effort,
    uses max_tokens instead of max_completion_tokens
  - OpenAI: requires real api_key, supports reasoning_effort
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("cuda_debugger_exp")


def detect_backend(config: dict) -> str:
    """Detect backend from config, with auto-detection heuristic."""
    explicit = config.get("api", {}).get("backend")
    if explicit:
        return explicit.lower()

    # Auto-detect: if base_url points to localhost / 127.0.0.1, likely vLLM
    base_url = config.get("api", {}).get("base_url") or ""
    if any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0")):
        return "vllm"

    return "openai"


def create_client(config: dict):
    """
    Create an OpenAI-compatible client from config.

    For vLLM backend: api_key defaults to "EMPTY" if not set.
    For OpenAI backend: api_key is required from environment.
    """
    import os
    from openai import OpenAI

    backend = detect_backend(config)
    api_cfg = config.get("api", {})

    # Resolve API key
    api_key_env = api_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")

    if backend == "vllm":
        # vLLM doesn't need a real key
        if not api_key:
            api_key = api_cfg.get("api_key", "EMPTY")
        base_url = api_cfg.get("base_url")
        if not base_url:
            # Default vLLM endpoint
            host = api_cfg.get("host", "localhost")
            port = api_cfg.get("port", 8000)
            base_url = f"http://{host}:{port}/v1"
        logger.info(f"Using vLLM backend: {base_url}, model={api_cfg.get('model', '?')}")
    else:
        if not api_key:
            raise RuntimeError(
                f"API key not found. Set {api_key_env} environment variable, "
                f"or use backend: vllm for local models."
            )
        base_url = api_cfg.get("base_url")
        logger.info(f"Using OpenAI backend: model={api_cfg.get('model', '?')}")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if backend == "vllm":
        # Disable httpx connection-pool keepalive for vLLM. vLLM occasionally
        # closes idle conns under load; pooled clients miss the FIN and
        # accumulate CLOSE_WAIT until the server stops accepting new requests.
        # Local TCP reconnect is sub-ms so the throughput cost is negligible.
        import httpx
        kwargs["http_client"] = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=0)
        )
    return OpenAI(**kwargs)


def call_llm(client, config: dict, messages: list[dict]) -> str:
    """
    Call the LLM and return the assistant message content.

    Adapts parameters based on backend:
      - openai: max_completion_tokens, reasoning_effort
      - vllm:   max_tokens, temperature (no reasoning_effort)
    """
    api_cfg = config["api"]
    backend = detect_backend(config)

    kwargs = dict(
        model=api_cfg["model"],
        messages=messages,
        # No HTTP timeout on the LLM call. The 300s in configs is for the
        # generated code's build/test (build.timeout / build.test_timeout),
        # not for model generation, which can legitimately run >5min on slow
        # local vLLM (Qwen3.6-27B single-GPU @ 16K tokens). Opt back in by
        # setting api.http_timeout if a hard cap is ever needed.
        timeout=api_cfg.get("http_timeout"),
    )

    if backend == "vllm":
        # vLLM uses max_tokens, not max_completion_tokens
        kwargs["max_tokens"] = api_cfg.get("max_tokens", api_cfg.get("max_completion_tokens", 4096))
        kwargs["temperature"] = api_cfg.get("temperature", 0.7)
        # vLLM-specific params
        if api_cfg.get("top_p") is not None:
            kwargs["top_p"] = api_cfg["top_p"]
        if api_cfg.get("repetition_penalty") is not None:
            kwargs["extra_body"] = {"repetition_penalty": api_cfg["repetition_penalty"]}
        if api_cfg.get("stop"):
            kwargs["stop"] = api_cfg["stop"]
    else:
        # OpenAI / compatible cloud API
        kwargs["max_completion_tokens"] = api_cfg.get("max_completion_tokens", api_cfg.get("max_tokens", 4096))
        reasoning_effort = api_cfg.get("reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = api_cfg.get("temperature", 0.7)
        if api_cfg.get("top_p") is not None:
            kwargs["top_p"] = api_cfg["top_p"]

    # Generic extra_body passthrough (e.g. NVIDIA-hosted models'
    # chat_template_kwargs for thinking control). Merges with anything
    # already set above (vLLM repetition_penalty).
    if api_cfg.get("extra_body"):
        kwargs.setdefault("extra_body", {}).update(api_cfg["extra_body"])

    # Per-call seed (used by repeated sampling — Exp 2). vLLM honours this;
    # most cloud providers ignore it (we record actual reproducibility in
    # _meta.seed_supported elsewhere).
    if api_cfg.get("seed") is not None:
        kwargs["seed"] = int(api_cfg["seed"])

    # Streaming path: needed for endpoints with strict gateway timeouts
    # (e.g. NVIDIA NIM 5-min cap on long thinking-mode generations).
    # Accumulates content + reasoning_content chunks separately.
    use_stream = bool(api_cfg.get("stream", False))

    def _one_call() -> tuple[str | None, str | None]:
        if use_stream:
            kwargs["stream"] = True
            cps: list[str] = []
            rps: list[str] = []
            for chunk in client.chat.completions.create(**kwargs):
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                c = getattr(delta, "content", None)
                if c:
                    cps.append(c)
                r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if r:
                    rps.append(r)
            return ("".join(cps) if cps else None, "".join(rps) if rps else None)
        else:
            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            u = getattr(response, "usage", None)
            if u is not None:
                # API-spend visibility (completion includes billed reasoning tokens)
                logger.info(
                    f"  API usage: prompt={getattr(u, 'prompt_tokens', '?')} "
                    f"completion={getattr(u, 'completion_tokens', '?')}"
                )
            return (msg.content, getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None))

    # Retry on empty content OR transient network/stream errors. NVIDIA-hosted
    # endpoints occasionally drop streams mid-flight (RemoteProtocolError) or
    # finish reasoning without emitting any content tokens.
    # api.retry_on_empty: int, default 2 extra attempts.
    import httpx
    transient_errors = (
        httpx.RemoteProtocolError,
        httpx.ReadError,
        httpx.ReadTimeout,
        httpx.ConnectError,
    )

    def _try_once_with_catch() -> tuple[str | None, str | None, Exception | None]:
        try:
            c, r = _one_call()
            return c, r, None
        except transient_errors as e:
            return None, None, e

    max_retries = int(api_cfg.get("retry_on_empty", 2))
    content, reasoning, last_err = _try_once_with_catch()
    attempts = 1
    while (not content) and attempts <= max_retries:
        why = (
            f"transient stream error: {type(last_err).__name__}"
            if last_err
            else f"empty content (reasoning={len(reasoning) if reasoning else 0} chars)"
        )
        logger.warning(f"LLM call failed/empty: {why}. Retrying ({attempts}/{max_retries})...")
        content, reasoning, last_err = _try_once_with_catch()
        attempts += 1
    if last_err and not content and not reasoning:
        # All retries exhausted — re-raise so run_experiment can record API error.
        raise last_err

    # Final fallback: if still empty, use reasoning as last resort (matches old behavior).
    if not content:
        if reasoning:
            logger.warning(
                f"LLM content still None after {attempts} attempts. "
                f"Falling back to reasoning ({len(reasoning)} chars) as content."
            )
            content = reasoning
        else:
            content = ""
            logger.warning(f"LLM returned empty response after {attempts} attempts (content+reasoning both empty)")

    return content


def extract_code(response_text: str) -> str:
    """Extract code from LLM response, stripping markdown fences if present."""
    text = response_text.strip()
    m = re.search(r"```(?:cuda|cpp|c\+\+|python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text
