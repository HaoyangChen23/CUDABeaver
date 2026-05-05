# CUDA-Debugger-Bench — Code Package

Companion code to the **CUDA-Debugger-Bench** dataset (213 broken CUDA
kernels). NeurIPS 2026 Datasets and Benchmarks Track — anonymous submission.

The code lets you (a) run any of the 6 evaluated LLMs against the benchmark
and reproduce the paper tables, (b) extend the harness with your own LLM, or
(c) re-curate / sub-sample the benchmark.

---

## Layout

```
harness/                   evaluation engine
  run_experiment.py          main entrypoint (gen → build → test → score)
  compiler.py                build dispatcher (nvcc / make / inline-cuda / class 2 runner)
  classifier.py              5-category outcome classifier
  feedback.py                Exp 3: feedback-level dispatcher (L0 … L4, error-aware modes)
  few_shot.py                Exp 7: in-context debug-pair retrieval
  benchmark.py               perf measurement orchestrator
  timing_modes/              5 timing backends (elapsed, kernels, applied-kernels, region, kernelbench)
  applied_kernels_eval.py    adapter for Class 2 def.py-based tasks
  applied_kernels_runner/    vendored class-1/2 backend (runner, eval, compiler, …)
  _vendor/kernelbench/       upstream KernelBench (MIT, ScalingIntelligence/KernelBench)
  llm_client.py              OpenAI-compatible cloud-API + vLLM client
  parallel_runner.py         per-task parallelism (N workers across N GPUs)
  queue_daemon.py            crash-safe queue daemon for long-batch sweeps
  references/schema.py       task / instance schema definitions

analysis/                  result aggregation
  scanner.py                 walks outputs/ and emits Run records
  pool.py                    per-task pooled metrics + 5-category counts
  axis_normalize.py          cfg basename → (model, axis, experiment) parser
  build_master.py            CLI: produces master.csv across all runs
  pivot_tables.py            CLI: paper tables + per-model summary + summary.md
  render_latex.py            booktabs LaTeX renderer

configs/                   238 evaluation configs across 5 axes
  iter/                      single-shot debug
  repeated/                  repeated@k vs iterative@k
  feedback/                  L0 (none) → L4 (oracle); + L3_raw + error-aware modes
  history/                   H1 … H4 multi-turn history rounds
  fewshot/                   in-context debug-pair retrieval

scripts/                   one-time setup + utilities
  prepare_benchmark.sh       materialize flat dataset layout (one-time, idempotent)
  post_conda_setup.sh        register vendored kernelbench in conda env (one-time)

benchmark/                 legacy task-list manifest (used by --mode debug)
docs/                      PROTOCOL, PERF_PROTOCOL, REPRODUCE, ANONYMIZATION
tests/                     smoke + unit tests (no LLM key required)
examples/                  reproduction examples (see docs/REPRODUCE.md)
```

---

## Get the dataset

The 213 broken-kernel instances live in a separate companion repository
(NeurIPS submission ships code + dataset as two URLs). Clone it as a
**sibling** of this repo so the default `prepare_benchmark.sh` path works:

```bash
# In the parent directory of this code package:
git clone https://huggingface.co/datasets/neurips26-anon/cuda-debugger-bench  neurips2026_dataset

# Or `huggingface-cli download`:
huggingface-cli download --repo-type dataset neurips26-anon/cuda-debugger-bench --local-dir neurips2026_dataset
```

Resulting layout:

```
parent/
├── neurips2026_code/        ← this repo
└── neurips2026_dataset/     ← cloned alongside (~117 MB)
    ├── data/                  213 instance subdirs + bundled CUTLASS/TK
    ├── manifest.json
    ├── croissant.json
    ├── README.md
    └── LICENSE
```

If you put the dataset somewhere else, pass its absolute path to
`prepare_benchmark.sh` (the script accepts any path). The dataset URL is
also reachable via the `Dataset URL` field of the OpenReview submission.

> Note for reviewers during double-blind review: the dataset URL above is
> the anonymous HuggingFace repo. The Croissant metadata file
> (`croissant.json`) is also attached as supplementary material on
> OpenReview per the D&B Track requirements.

---

## Setup

### Option A — pip + venv (recommended)

```bash
bash setup_env.sh
source .venv/bin/activate
bash scripts/prepare_benchmark.sh ../neurips2026_dataset
```

`setup_env.sh` creates a venv from `requirements.txt` and registers the
vendored `kernelbench` package via a `.pth` so `import kernelbench` works.
`prepare_benchmark.sh` materializes the flat dataset layout the harness
expects (idempotent — re-runnable any time).

### Option B — conda

```bash
conda env create -f environment.yml
conda activate cuda-debugger-bench
bash scripts/post_conda_setup.sh
bash scripts/prepare_benchmark.sh ../neurips2026_dataset
```

`environment.yml` pulls PyTorch + CUDA from the official `pytorch` and
`nvidia` channels for ABI alignment with the system `nvcc`. Adjust
`pytorch-cuda` if your toolkit version (12.x vs 13.x) differs.

### System prerequisites

- CUDA toolkit (`nvcc` ≥ 12.0; CUDA 13 is supported — see
  `docs/REPRODUCE.md` for the thrust-path note)
- A CUDA-capable GPU (SM ≥ 80)
- Python 3.10–3.12 (3.11 is the tested target)

---

## Smoke test (no LLM key required)

```bash
pytest tests/test_smoke.py -v
```

Validates harness imports + classifier + bundled fixture instance.
Should complete in under 1 s.

---

## End-to-end reproduction

See **`docs/REPRODUCE.md`** for the full command sequence per paper table.
Quick sketch (Table 1, iter axis):

```bash
for cfg in configs/iter/*.yaml; do
  python -m harness.run_experiment \
    --config "$cfg" \
    --v2-task-list benchmark/task_list.json \
    --mode debug
done

python -m analysis.build_master  --results-root outputs --out-dir analysis_out
python -m analysis.pivot_tables  --in-dir analysis_out --out-dir analysis_out
```

`analysis_out/tables/exp1_iter.tex` matches paper Table 1.

### LLM endpoints — required swap

Each cfg's `llm:` block ships with placeholder `<CLOUD_API_BASE_URL>` +
`CLOUD_API_KEY`. Reviewers must point these at their own infrastructure
(any OpenAI-compatible cloud API or a self-hosted vLLM server). See
`docs/REPRODUCE.md § LLM endpoints` for details.

---

## Family-specific runtime requirements

All required code/headers are bundled — most instances build with just
`nvcc`. Per-family detail:

| Family                                  | n  | Bundled at                                          |
|-----------------------------------------|----|-----------------------------------------------------|
| cuda, cublas, cufft, cusolver, cusparse | 70 | (nvcc only)                                         |
| cutlass                                 | 20 | `../neurips2026_dataset/data/_external/cutlass`     |
| thunderkittens                          |  4 | testbench self-contained + bundled fallback         |
| kernelbench                             | 84 | `harness/_vendor/kernelbench/` (registered via .pth)|
| flashattention, layernorm, cufft_samples| 35 | `harness/applied_kernels_runner/` + reference_sources |

**Total: 213 instances**, 100% buildable on CUDA 12 or 13 (see the
build-audit table in `docs/REPRODUCE.md`).

---

## What's NOT in the package

- Paper-side `outputs/` directories (reviewer-side, regenerated per run)
- API keys / `.env` / scratch caches
- The dataset itself (separate package: see `benchmark/README.md`)
- Profiler dumps (`*.nsys-rep`, `*.ncu-rep`)

---

## Authors

Anonymous Authors (NeurIPS 2026 D&B Track submission). Will be deanonymized
on acceptance.

## License

The harness + analysis code is released under **CC BY 4.0** (see
`LICENSE`). Vendored third-party packages keep their own licenses:

- `harness/_vendor/kernelbench/` — MIT, see
  `harness/_vendor/kernelbench/LICENSE.txt`

The dataset's bundled external libraries (CUTLASS, ThunderKittens) keep
their respective BSD-3-Clause / MIT licenses; see the
`data/_external/<lib>/LICENSE.txt` files in the dataset package.
