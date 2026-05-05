# Benchmark Dataset

The companion CUDA-Debugger-Bench dataset is published separately on
HuggingFace:

    https://huggingface.co/datasets/anonymous-neurips2026/cuda-debugger-bench

Download with `huggingface-cli download` (or `datasets.load_dataset`) and
point the harness at the resulting dataset directory.

## task_list.json

`task_list.json` in this directory is a legacy-format task list reconstructed
from the dataset package's `manifest.json` and per-instance `instance.json`
files. It is consumed by `harness.run_experiment --mode debug` via the
`--v2-task-list` argument to look up each cfg-stem's `broken_start` metadata.
See `docs/REPRODUCE.md` for the exact command.
