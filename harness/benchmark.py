"""Performance benchmark pipeline.

Loads references from evaluation/references/{stem}.json, runs them and
candidates, caches reference timings per (gpu, cuda_version) signature.
"""
from __future__ import annotations
import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import shutil
import tempfile

from .references.schema import Reference, ReferenceSolution

ROOT = Path(__file__).resolve().parents[2]
REFS_DIR = ROOT / "evaluation/references"
CACHE_DIR = ROOT / "evaluation/references_cache"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


@lru_cache(maxsize=1)
def gpu_signature() -> str:
    """Return a stable string identifying (GPU model, sm, cuda toolkit)."""
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits", "-i", "0"],
        text=True,
    ).strip()
    cap = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits", "-i", "0"],
        text=True,
    ).strip()
    sm = "sm_" + cap.replace(".", "")
    nvcc_out = subprocess.check_output(["nvcc", "--version"], text=True)
    cuda_match = re.search(r"release (\d+)\.(\d+)", nvcc_out)
    cuda = (
        f"cuda_{cuda_match.group(1)}_{cuda_match.group(2)}"
        if cuda_match
        else "cuda_unknown"
    )
    return f"{_slug(name)}_{sm}_{cuda}"


@dataclass
class CacheEntry:
    task_id: str
    reference_hash: str
    gpu_signature: str
    cuda_version: str
    timing: dict
    measured_at: str
    elapsed_total_s: float
    skipped: Optional[str] = None  # e.g. "incompatible_arch"


def task_id_to_stem(task_id: str) -> str:
    return task_id.replace("/", "_")


def load_reference(task_id: str) -> Reference:
    stem = task_id_to_stem(task_id)
    path = REFS_DIR / f"{stem}.json"
    return Reference.from_dict(json.load(open(path)))


def _cache_path(task_id: str) -> Path:
    sig = gpu_signature()
    return CACHE_DIR / sig / f"{task_id_to_stem(task_id)}.json"


def cache_lookup(ref: Reference) -> Optional[CacheEntry]:
    p = _cache_path(ref.task_id)
    if not p.exists():
        return None
    d = json.load(open(p))
    if d.get("reference_hash") != ref.canonical_hash():
        return None
    if d.get("gpu_signature") != gpu_signature():
        return None
    return CacheEntry(**d)


def cache_write(entry: CacheEntry) -> None:
    p = _cache_path(entry.task_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(entry), indent=2) + "\n")


def prepare_workdir(task_id: str, solution: ReferenceSolution) -> Path:
    """Create a fresh workdir: copy testbench, materialize solution files.

    Returns the workdir path. This module never deletes — caller is responsible
    for cleanup if explicitly authorized.
    """
    stem = task_id_to_stem(task_id)
    testbench_src = ROOT / "testbench" / stem
    workdir = Path(tempfile.mkdtemp(prefix=f"benchmark_{stem}_"))

    if testbench_src.is_dir():
        shutil.copytree(testbench_src, workdir, dirs_exist_ok=True)

    if solution.type == "file":
        for f in solution.files:
            tgt = workdir / f["path"]
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(f["content"])
    elif solution.type == "kernelbench_problem":
        # Prefer embedded problem_content (self-contained); fall back to file
        if solution.problem_content is not None:
            problem_text = solution.problem_content
        else:
            problem_text = (ROOT / solution.problem_path).read_text()
        (workdir / "problem.py").write_text(problem_text)
        # For reference timing, solution.py mirrors problem with a ModelNew alias
        # so kernelbench.eval_kernel_against_ref accepts it and runtime ≈ ref_runtime
        solution_alias = problem_text + "\n\nModelNew = Model\n"
        (workdir / "solution.py").write_text(solution_alias)
    return workdir


import datetime  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Tuple  # noqa: E402

PYTHON_BIN = "/mnt/data/anonymous/envs/cuda-debugger-exp/bin/python"


def _build_env(gpu_id: int) -> dict:
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pybin = Path(PYTHON_BIN).parent
    env["PATH"] = f"{pybin}:{env.get('PATH','')}"
    return env


def _run_pg(cmd: str, *, cwd, env, timeout: int):
    """Run cmd in a new process group. On timeout, SIGKILL the whole group
    so grandchildren (kernelbench eval python under sh -c) don't orphan.
    Returns subprocess.CompletedProcess-like with returncode/stdout/stderr.
    """
    proc = subprocess.Popen(
        cmd, shell=True, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", "killed (process group timeout)"
        raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err)


def _execute_benchmark(ref: Reference, workdir: Path, env: dict) -> Tuple[dict, float]:
    """Run benchmark_command in workdir, parse with the timing-mode adapter.

    Two-phase: build first via `build_command` from the matching input/{stem}.json
    if such a build is required (curated_cuda_pool needs nvcc; kernelbench is run-only).
    Then wrap the benchmark_command per the timing-mode adapter (e.g. nsys for region).
    """
    from .timing_modes import get_timing_mode

    # Per-task isolated torch_extensions cache. LLM solutions frequently use
    # generic load_inline names (e.g. name="fused_ops"). Sharing the global
    # ~/.cache/torch_extensions/ across tasks causes ninja-lock contention
    # AND can return a stale .so from another task — both have happened.
    ext_dir = workdir / ".torch_ext"
    ext_dir.mkdir(exist_ok=True)
    env = dict(env)
    env["TORCH_EXTENSIONS_DIR"] = str(ext_dir)

    parser = get_timing_mode(ref.timing_mode)

    # Phase 1: build (if input/{stem}.json has a build_command). Skip for
    # kernelbench tasks since their build_command is a static checker that
    # rejects valid PyTorch reference baselines; the kernelbench eval itself
    # does any required JIT compilation when called.
    if ref.timing_mode.type != "kernelbench_eval_result":
        stem = task_id_to_stem(ref.task_id)
        input_json = ROOT / "input" / f"{stem}.json"
        if input_json.exists():
            meta = json.loads(input_json.read_text())
            build_cmd = meta.get("build_command")
            if build_cmd:
                build_cmd = re.sub(r"\bpython\b(?!3)", PYTHON_BIN, build_cmd)
                bp = _run_pg(build_cmd, cwd=workdir, env=env, timeout=300)
                if bp.returncode != 0:
                    raise RuntimeError(
                        f"build failed for {ref.task_id}: rc={bp.returncode}\n"
                        f"stderr_tail: {bp.stderr[-500:]}"
                    )

    # Phase 2: benchmark
    cmd = ref.benchmark_command
    cmd = re.sub(r"\bpython\b(?!3)", PYTHON_BIN, cmd)
    cmd = parser.wrap_command(cmd, str(workdir))

    # Per-GPU perf serialization. Multiple daemon children sharing one GPU
    # for nsys-instrumented perf collide on the CUDA driver mutex + nsys
    # capture lock — observed pathologically with 4× CUDA_87 thrust scans on
    # GPU 6 holding GPU at 0% util while pegging CPU. flock the cmd by
    # CUDA_VISIBLE_DEVICES so only one perf process at a time per physical GPU.
    gpu_for_lock = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    lock_path = f"/tmp/cuda_debugger_perf_lock_gpu{gpu_for_lock}.lock"
    import shlex
    cmd = f"flock {shlex.quote(lock_path)} sh -c {shlex.quote(cmd)}"

    t0 = time.time()
    proc = _run_pg(cmd, cwd=workdir, env=env, timeout=1800)
    elapsed = time.time() - t0

    timing = parser.parse(stdout=proc.stdout, stderr=proc.stderr)
    timing.setdefault("n_trials", ref.n_trials)
    timing.setdefault("warmup_trials", ref.warmup_trials)
    return timing, elapsed


def _arch_compatible(ref: Reference) -> bool:
    if ref.requires_arch is None:
        return True
    sig = gpu_signature()
    needed = ref.requires_arch.replace("a", "")  # sm_90a -> sm_90
    return needed in sig


def _now_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _cuda_version_from_signature() -> str:
    m = re.search(r"cuda_(\d+_\d+)", gpu_signature())
    return m.group(1).replace("_", ".") if m else ""


def run_reference(task_id: str, gpu_id: int = 0) -> CacheEntry:
    ref = load_reference(task_id)
    cached = cache_lookup(ref)
    if cached is not None:
        return cached

    if not _arch_compatible(ref):
        entry = CacheEntry(
            task_id=task_id,
            reference_hash=ref.canonical_hash(),
            gpu_signature=gpu_signature(),
            cuda_version="",
            timing={},
            measured_at=_now_iso(),
            elapsed_total_s=0.0,
            skipped="incompatible_arch",
        )
        cache_write(entry)
        return entry

    workdir = prepare_workdir(task_id, ref.reference_solution)
    env = _build_env(gpu_id)
    timing, elapsed = _execute_benchmark(ref, workdir, env)
    entry = CacheEntry(
        task_id=task_id,
        reference_hash=ref.canonical_hash(),
        gpu_signature=gpu_signature(),
        cuda_version=_cuda_version_from_signature(),
        timing=timing,
        measured_at=_now_iso(),
        elapsed_total_s=elapsed,
        skipped=None,
    )
    cache_write(entry)
    return entry


@dataclass
class CandidateTiming:
    task_id: str
    timing: dict
    elapsed_total_s: float
    measured_at: str


def run_candidate(
    task_id: str,
    candidate_files: list,
    gpu_id: int = 0,
    n_runs: int = 1,
) -> CandidateTiming:
    """Build + time the candidate. When n_runs > 1, repeat the timing
    (each run still does its own internal 100 trials) and compute CV
    across the n_runs mean values. CV ('coefficient of variation') is
    reported as `cv` in the timing dict; absent when n_runs == 1 or the
    parser already reports stddev_ms > 0.
    """
    ref = load_reference(task_id)
    if not _arch_compatible(ref):
        return CandidateTiming(
            task_id=task_id,
            timing={"skipped": "incompatible_arch"},
            elapsed_total_s=0.0,
            measured_at=_now_iso(),
        )
    cand_solution = ReferenceSolution(type="file", files=candidate_files)
    workdir = prepare_workdir(task_id, cand_solution)
    env = _build_env(gpu_id)

    n_runs = max(1, int(n_runs))
    if n_runs == 1:
        timing, elapsed = _execute_benchmark(ref, workdir, env)
        # If parser reported a real stddev, derive CV
        mean = timing.get("mean_ms")
        std = timing.get("stddev_ms")
        if isinstance(mean, (int, float)) and isinstance(std, (int, float)) and mean > 0 and std > 0:
            timing.setdefault("cv", std / mean)
        timing.setdefault("n_runs", 1)
        return CandidateTiming(
            task_id=task_id, timing=timing, elapsed_total_s=elapsed,
            measured_at=_now_iso(),
        )

    # n_runs > 1 — repeat the full measurement; each run already does
    # n_trials internal trials. CV is across the n_runs means.
    means: list = []
    last_timing = None
    total_elapsed = 0.0
    for _ in range(n_runs):
        timing, elapsed = _execute_benchmark(ref, workdir, env)
        total_elapsed += elapsed
        last_timing = timing
        m = timing.get("mean_ms")
        if isinstance(m, (int, float)) and m > 0:
            means.append(m)

    if not means or last_timing is None:
        # All runs failed timing — return whatever last gave us
        return CandidateTiming(
            task_id=task_id, timing=last_timing or {"skipped": "no_timing"},
            elapsed_total_s=total_elapsed, measured_at=_now_iso(),
        )

    import statistics
    mean = statistics.fmean(means)
    std = statistics.stdev(means) if len(means) >= 2 else 0.0
    last_timing["mean_ms"] = mean
    last_timing["stddev_ms"] = std
    last_timing["cv"] = (std / mean) if mean > 0 else float("nan")
    last_timing["n_runs"] = n_runs
    last_timing["per_run_means_ms"] = means
    return CandidateTiming(
        task_id=task_id, timing=last_timing,
        elapsed_total_s=total_elapsed, measured_at=_now_iso(),
    )


def compute_speedup(reference_mean_ms: float, candidate_mean_ms: float) -> float:
    if candidate_mean_ms <= 0:
        return float("nan")
    return reference_mean_ms / candidate_mean_ms
