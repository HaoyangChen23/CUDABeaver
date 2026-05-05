"""
Eight-category classifier for CUDA build/test results.
Aligned with CUDA-Debugger benchmark taxonomy (docs/error_taxonomy.md).

Categories (mutually exclusive, checked in priority order):
  1. environment_dependency_bug — GPU arch mismatch, driver issues, missing system libs
  2. integration_bug            — wrong file layout, harness contract mismatch
  3. compile_error              — code-intrinsic build failure (syntax, type, API misuse)
  4. out_of_memory              — cudaMalloc / OOM at runtime
  5. illegal_memory_access      — segfault, illegal address, double free
  6. timeout                    — execution exceeded time limit
  7. functional_correctness_bug — compiled & ran, wrong result
  8. passed                     — all checks passed
"""

import re

# --- Environment / GPU arch mismatch (checked FIRST to avoid misclassification) ---
ENVIRONMENT_PATTERNS = [
    r"no kernel image is available for execution on the device",
    r"cudaErrorNoKernelImageForDevice",
    r"Failed to query occupancy",
    r"CUDA driver version is insufficient",
    r"cudaErrorInsufficientDriver",
    r"CUDA_ERROR_NO_DEVICE",
    r"CUDA_ERROR_INVALID_DEVICE",
    r"GPU arch mismatch",
    r"the provided PTX was compiled with an unsupported toolchain",
]

# --- Illegal memory access (runtime crash) ---
ILLEGAL_MEMORY_PATTERNS = [
    r"Segmentation fault",
    r"SIGSEGV",
    r"illegal memory access",
    r"cudaErrorIllegalAddress",
    r"double free",
    r"an illegal memory access was encountered",
]

# --- Out of memory ---
OOM_PATTERNS = [
    r"out of memory",
    r"cudaErrorMemoryAllocation",
    r"CUDA_ERROR_OUT_OF_MEMORY",
    r"cudaMalloc failed",
]

# --- Timeout ---
TIMEOUT_PATTERNS = [
    r"timed out",
    r"SIGKILL",
    r"killed",
    r"\[TEST TIMEOUT",
    r"Time limit exceeded",
]


def classify_result(
    build_ok: bool,
    build_output: str,
    test_output: str | None,
    test_returncode: int | None,
) -> dict:
    """
    Classify a single iteration result into one of eight categories.

    Returns dict with keys:
      - five_category: str   (kept for backward compat, maps 8-cat to 5-cat)
      - category: str        (8-category label)
      - error_type: str | None
      - build_ok: bool
      - status: str          (failed | passed)
      - logic_error_category: str | None
      - logic_error_detail: str | None
    """
    combined = (build_output or "") + "\n" + (test_output or "")

    # --- 1. Environment dependency (check both build and test output) ---
    env_match = _match_patterns(combined, ENVIRONMENT_PATTERNS)
    if env_match:
        return _result(
            category="environment_dependency_bug",
            five_category="logic_error",  # legacy compat
            error_type="environment",
            build_ok=build_ok,
            detail=f"Environment: {env_match}",
        )

    # --- 2. Build failure ---
    if not build_ok:
        # Always classify as compile error when build failed,
        # regardless of whether specific error patterns match.
        # _has_compile_error may miss non-C++ errors (e.g., Python SyntaxError).
        return _result(
            category="buildability_bug",
            five_category="compile_error",
            error_type="compile_only",
            build_ok=False,
            detail=build_output[:500] if build_output else None,
        )

    # --- 3. Build succeeded, check test ---
    if test_returncode == 0 and test_output is not None:
        if (not _match_patterns(test_output, ILLEGAL_MEMORY_PATTERNS)
                and not _match_patterns(test_output, OOM_PATTERNS)
                and not _match_patterns(test_output, TIMEOUT_PATTERNS)):
            # applied-kernels-aware perf gate: if test_output contains "Speedup:" lines
            # (applied-kernels ./test format), require mean_speedup >= 0.5x CUTLASS to
            # actually count as passed. correctness alone (applied-kernels naive baseline)
            # would otherwise pass trivially without any optimization.
            kh_speedups = _extract_kh_speedups(test_output)
            if kh_speedups:
                import statistics
                mean_sp = statistics.mean(kh_speedups)
                if mean_sp < 0.5:
                    return _result(
                        category="perf_below_threshold",
                        five_category="logic_error",
                        error_type="perf_gate",
                        build_ok=True,
                        detail=f"correctness OK but mean speedup {mean_sp:.3f}x < 0.5x CUTLASS reference (per-size: {kh_speedups})",
                    )
                # passed-with-perf: include speedup numbers in detail for D-report
                return _result(
                    category="passed",
                    five_category="passed",
                    build_ok=True,
                    status="passed",
                    detail=f"mean_speedup={mean_sp:.3f}x  per-size={kh_speedups}",
                )
            return _result(
                category="passed",
                five_category="passed",
                build_ok=True,
                status="passed",
            )

    test_output = test_output or ""

    # --- 4. OOM ---
    oom_match = _match_patterns(test_output, OOM_PATTERNS)
    if oom_match:
        return _result(
            category="out_of_memory",
            five_category="memory_crash",
            error_type="test_only",
            build_ok=True,
            detail=f"OOM: {oom_match}",
        )

    # --- 5. Illegal memory access ---
    mem_match = _match_patterns(test_output, ILLEGAL_MEMORY_PATTERNS)
    if mem_match:
        return _result(
            category="illegal_memory_access",
            five_category="memory_crash",
            error_type="test_only",
            build_ok=True,
            detail=f"Memory: {mem_match}",
        )

    # --- 6. Timeout ---
    if _match_patterns(test_output, TIMEOUT_PATTERNS):
        return _result(
            category="timeout",
            five_category="timeout",
            error_type="test_only",
            build_ok=True,
            detail="Execution timed out",
        )

    # --- 7. Functional correctness bug ---
    return _result(
        category="functional_correctness_bug",
        five_category="logic_error",
        error_type="test_only",
        build_ok=True,
        detail=test_output[:500] if test_output else "No test output",
    )


def _result(
    category: str,
    five_category: str,
    build_ok: bool = False,
    error_type: str | None = None,
    status: str = "failed",
    detail: str | None = None,
) -> dict:
    return {
        "category": category,
        "five_category": five_category,
        "error_type": error_type,
        "build_ok": build_ok,
        "status": status,
        "logic_error_category": category if status == "failed" else None,
        "logic_error_detail": detail,
    }


def _extract_kh_speedups(output: str) -> list[float]:
    """Extract `Speedup: X.XXXXx` numbers from applied-kernels ./test stdout.
    Returns empty list if no Speedup lines found (non-applied-kernels task).
    """
    return [float(m) for m in re.findall(r"Speedup:\s*([\d.]+)x", output or "")]


def _match_patterns(output: str, patterns: list[str]) -> str | None:
    """Return the first matching line, or None."""
    for line in output.splitlines():
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return line.strip()[:200]
    return None


def _has_compile_error(output: str) -> bool:
    for line in output.splitlines():
        if re.search(r"\berror\b", line, re.IGNORECASE):
            if not re.search(r"\bwarning\b", line, re.IGNORECASE):
                return True
    return False
