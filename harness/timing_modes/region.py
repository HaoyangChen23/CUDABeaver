"""Parse `nsys profile --stats=true` NVTX region (nvtx_sum) output.

Sample relevant section:
    [3/8] Executing 'nvtx_sum' stats report

     Time (%)  Total Time (ns)  Instances    Avg (ns)   ...   Style       Range
     --------  ---------------  ---------  -----------       -------  -------------
        100.0      286,871,681          1  286,871,681.0     PushPop  :bench_region

Numbers may have commas; the Range column is prefixed with ':' (NVTX domain).
The 'Avg (ns)' column is column index 3 (0-based) of the data row.
"""
import re


class NvtxRegion:
    def __init__(self, include, time_type):
        self.include = include or []
        self.time_type = time_type

    def wrap_command(self, cmd: str, workdir: str) -> str:
        # Run inside nsys with stats; TMPDIR override avoids /tmp/nvidia perm issue
        return (
            f"TMPDIR={workdir} nsys profile --stats=true --force-overwrite=true "
            f"-o {workdir}/_nvtx_cap {cmd}"
        )

    def parse(self, stdout: str, stderr: str) -> dict:
        text = stdout + "\n" + stderr
        if not self.include:
            raise ValueError("nvtx_region requires include=[regions]")
        target = self.include[0]
        # Match a row whose final column ends with ":<target>" or is exactly "<target>"
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("-"):
                continue
            # Skip header rows (containing "Time (%)" or "Avg (ns)")
            if "Time (%)" in stripped or "Avg (ns)" in stripped:
                continue
            tokens = stripped.split()
            if len(tokens) < 6:
                continue
            range_token = tokens[-1].lstrip(":")
            if range_token != target:
                continue
            try:
                # Standard nvtx_sum row: Time%, Total, Instances, Avg, Med, Min, Max, StdDev, Style, Range
                # Tokens length 10. Avg is index 3. Instances is index 2.
                avg_ns_str = tokens[3].replace(",", "")
                inst_str = tokens[2].replace(",", "")
                avg_ns = float(avg_ns_str)
                instances = int(inst_str)
            except (ValueError, IndexError):
                continue
            mean_ms = avg_ns / 1e6
            return {
                "method": f"region/{target}",
                "mean_ms": mean_ms,
                "p50_ms": mean_ms,
                "p95_ms": mean_ms,
                "stddev_ms": 0.0,
                "raw_samples_ms": [],
                "n_instances": instances,
            }
        raise ValueError(
            f"NVTX region {target!r} not found in nsys output:\n{text[-500:]}"
        )
