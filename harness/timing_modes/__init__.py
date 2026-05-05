"""Timing-mode parsers for benchmark.py."""
from typing import Protocol

from ..references.schema import TimingMode


class TimingParser(Protocol):
    def wrap_command(self, cmd: str, workdir: str) -> str:
        """Optionally wrap the benchmark command (e.g. with nsys profile)."""
        ...

    def parse(self, stdout: str, stderr: str) -> dict: ...


def get_timing_mode(mode: TimingMode) -> TimingParser:
    if mode.type == "region":
        from .region import NvtxRegion
        return NvtxRegion(include=mode.include or [], time_type=mode.time_type or "kernel")
    if mode.type == "kernels":
        from .kernels import CudaKernelsAggregate
        return CudaKernelsAggregate()
    if mode.type == "process":
        from .elapsed import TotalElapsed
        return TotalElapsed()  # process == wall-clock, reuse total_elapsed
    if mode.type == "kernelbench_eval_result":
        from .kernelbench import KernelbenchEvalResult
        return KernelbenchEvalResult()
    if mode.type == "total_elapsed":
        from .elapsed import TotalElapsed
        return TotalElapsed()
    if mode.type == "applied_kernels_make_speedup":
        from .applied_kernels_make import AppliedKernelsMakeSpeedup
        return AppliedKernelsMakeSpeedup()
    raise ValueError(f"unknown timing_mode.type: {mode.type}")
