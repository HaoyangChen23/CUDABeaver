# Reproducing Paper Tables

## One-time setup

The dataset (213 instances + bundled CUTLASS / ThunderKittens headers) ships
as a separate repository. Clone it as a sibling of this code package, then
run the setup scripts:

```bash
# Step 1 — clone the dataset alongside this repo.
cd ..   # parent of neurips2026_code/
git clone https://huggingface.co/datasets/neurips26-anon/cuda-debugger-bench  neurips2026_dataset
cd neurips2026_code/

# Step 2 — set up the python env + register vendored kernelbench (.pth).
bash setup_env.sh
source .venv/bin/activate

# Step 3 — symlink the dataset's flat layout into <code_root>/data/.
bash scripts/prepare_benchmark.sh ../neurips2026_dataset
```

`setup_env.sh` creates a venv, installs `requirements.txt`, and registers the
vendored upstream `kernelbench` package (`harness/_vendor/kernelbench/`,
MIT-licensed, see `harness/_vendor/kernelbench/LICENSE.txt`) onto the venv's
`site-packages` via a `.pth` file so `import kernelbench` works.

`prepare_benchmark.sh` materializes a flat `data/{input,prompts,testbench,
references}/` symlink layout and adds per-testbench `reference_sources/`
symlinks pointing into the bundled CUTLASS / ThunderKittens headers under
`../neurips2026_dataset/data/_external/`. It is idempotent.

## LLM endpoints — REQUIRED swap

Each cfg's `llm:` block has two fields a reviewer must point at their own
infrastructure before running the experiment:

- `base_url` — an OpenAI-compatible HTTPS endpoint. The harness uses any
  cloud API that follows the OpenAI Chat Completions schema, or a self-hosted
  vLLM server (`http://<host>:<port>/v1`).
- `api_key_env` — name of the env var that holds the API key. For local vLLM
  serving, set this var to any non-empty placeholder.

Configs ship with the endpoints + env-var names we used to generate the paper
results. Edit them, or use a launcher that overrides the cfg, to point at your
own cloud-API account or vLLM server before running. Configs targeting a local
vLLM server will silently fail until reachable.

## Family-specific runtime requirements

Most instances build with just `nvcc` from the CUDA toolkit. A few families
need additional setup; all required code/headers are bundled:

| Family                                  | n  | Bundled at                                          |
|-----------------------------------------|----|-----------------------------------------------------|
| cuda, cublas, cufft, cusolver, cusparse | 70 | (nvcc only)                                         |
| cutlass                                 | 20 | `../neurips2026_dataset/data/_external/cutlass`     |
| thunderkittens                          |  4 | testbench self-contained + bundled fallback         |
| kernelbench                             | 84 | `harness/_vendor/kernelbench/` (registered via .pth)|
| flashattention, layernorm, cufft_samples| 35 | `harness/applied_kernels_runner/` + reference_sources |

## Reproducing tables

The harness's `--mode debug` path needs a legacy-format task list that maps
each cfg-side stem to its `broken_start` metadata. We ship one regenerated
from the public dataset at `benchmark/task_list.json` (see
`benchmark/README.md` for how it was built).

## Table 1 (iter axis, 7 models)

```bash
for cfg in configs/iter/*.yaml; do
  python -m harness.run_experiment \
    --config "$cfg" \
    --v2-task-list benchmark/task_list.json \
    --mode debug
done
python -m analysis.build_master --results-root outputs/iter --out-dir analysis_out
python -m analysis.pivot_tables --master analysis_out/master.csv --out-dir analysis_out
```

Output: `analysis_out/table_iter.tex` matches paper Table 1.

(Repeat for axes: repeated, feedback, history, fewshot.)
