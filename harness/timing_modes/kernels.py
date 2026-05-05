"""All-CUDA-kernels aggregate parser.

Parses `nsys profile --stats=true` `cuda_gpu_kern_sum` table and SUMS
'Total Time (ns)' across all kernels. Use --cuda-graph-trace=node so
graph-launched kernels show up too (CUDA/8 etc).

Sample input:
    [6/8] Executing 'cuda_gpu_kern_sum' stats report

     Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)   Name
     --------  ---------------  ---------  --------  --------  --------  --------  -----------   --------
         22.2      155,092,333     10,300  15,057.5  12,032.0     9,728  2,096,869     76,769.1   normalize_image(...)
         ...

Reported metric: SUM of Total Time (ns) -> mean_ms = sum_ns / 1e6.
This is "total kernel work in the benchmark run". Speedup ratio
(ref/candidate) is the meaningful comparison.
"""
import re


class CudaKernelsAggregate:
    def wrap_command(self, cmd: str, workdir: str) -> str:
        # --cuda-graph-trace=node ensures graph-launched kernels are tracked
        return (
            f"TMPDIR={workdir} nsys profile --stats=true --cuda-graph-trace=node "
            f"--force-overwrite=true -o {workdir}/_kern_cap {cmd}"
        )

    def parse(self, stdout: str, stderr: str) -> dict:
        text = stdout + "\n" + stderr
        # Find the cuda_gpu_kern_sum section
        m = re.search(
            r"Executing 'cuda_gpu_kern_sum'.*?(?=Executing |Generated:|SKIPPED:|$)",
            text,
            re.DOTALL,
        )
        if not m:
            raise ValueError(
                f"cuda_gpu_kern_sum section not found in nsys stdout:\n{text[-500:]}"
            )
        section = m.group(0)

        # Each data row: Time%, Total(ns), Instances, Avg, Med, Min, Max, StdDev, Name
        # Numbers may have commas. The Name column may have spaces.
        total_ns_sum = 0.0
        n_kernels = 0
        n_instances = 0
        for line in section.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("-"):
                continue
            if "Time (%)" in stripped or "Executing" in stripped:
                continue
            tokens = stripped.split()
            if len(tokens) < 8:
                continue
            try:
                total_ns = float(tokens[1].replace(",", ""))
                instances = int(tokens[2].replace(",", ""))
            except (ValueError, IndexError):
                continue
            total_ns_sum += total_ns
            n_kernels += 1
            n_instances += instances

        if n_kernels == 0:
            raise ValueError(
                f"cuda_gpu_kern_sum section had no parseable rows:\n{section[-500:]}"
            )

        mean_ms = total_ns_sum / 1e6
        return {
            "method": "cuda_kernels_aggregate",
            "mean_ms": mean_ms,
            "p50_ms": mean_ms,
            "p95_ms": mean_ms,
            "stddev_ms": 0.0,
            "raw_samples_ms": [],
            "n_kernels": n_kernels,
            "n_kernel_launches": n_instances,
        }
