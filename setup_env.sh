#!/usr/bin/env bash
# Set up Python venv with pinned requirements + register vendored packages,
# and verify the CUDA toolchain the harness depends on.
#
# After this completes, activate with: `source .venv/bin/activate`
# and re-run prepare_benchmark.sh to materialize symlinks.
set -e

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# CUDA toolchain checks (fail early, not 3 hours into a run)
# ---------------------------------------------------------------------------
# The corpus needs CUDA >= 12.8: cusolver tasks call cusolverDnXgeev
# (introduced in 12.8) and several testbenches include <nvtx3/nvtx3.hpp>.
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not on PATH. Export your toolkit first, e.g.:"
    echo "  export PATH=/usr/local/cuda-12.9/bin:\$PATH"
    echo "  export CUDA_HOME=/usr/local/cuda-12.9"
    exit 1
fi
NVCC_VER="$(nvcc --version | grep -o 'release [0-9]*\.[0-9]*' | grep -o '[0-9.]*')"
NVCC_MAJOR="${NVCC_VER%%.*}"; NVCC_MINOR="${NVCC_VER##*.}"
if [ "$NVCC_MAJOR" -lt 12 ] || { [ "$NVCC_MAJOR" -eq 12 ] && [ "$NVCC_MINOR" -lt 8 ]; }; then
    echo "ERROR: nvcc $NVCC_VER found, but the corpus requires CUDA >= 12.8"
    echo "(cusolverDnXgeev, nvtx3). Point PATH/CUDA_HOME at a newer toolkit."
    exit 1
fi
if [ -z "${CUDA_HOME:-}" ]; then
    echo "WARNING: CUDA_HOME is not set. torch's JIT extension builds resolve"
    echo "nvcc via CUDA_HOME first and may pick up a stale system toolkit."
    echo "Recommended:  export CUDA_HOME=\"\$(dirname \"\$(dirname \"\$(command -v nvcc)\")\")\""
fi
echo "nvcc $NVCC_VER OK"

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Register vendored packages (KernelBench upstream MIT-licensed pkg) into the
# venv's site-packages via a .pth file. This makes `import kernelbench` work
# after activation without editing PYTHONPATH manually.
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
echo "$CODE_ROOT/harness/_vendor" > "$SITE_PACKAGES/applied_kernels_vendored.pth"

# Runtime sanity: torch must see the GPU (a CUDA-variant mismatch with the
# driver is the most common failure), and ninja must be importable.
python3 - <<'PYEOF'
import sys
import torch, ninja  # noqa: F401
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False — your torch wheel's "
             "CUDA version likely exceeds the driver's. Reinstall the matching "
             "variant, e.g. pip install torch --index-url "
             "https://download.pytorch.org/whl/cu128")
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) sees "
      f"{torch.cuda.device_count()} GPU(s); ninja OK")
PYEOF

echo
echo "Setup complete."
echo "  - venv:           $CODE_ROOT/.venv"
echo "  - vendored .pth:  $SITE_PACKAGES/applied_kernels_vendored.pth"
echo
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  bash scripts/prepare_benchmark.sh"
echo "  pytest tests/test_smoke.py"
echo
echo "Run the harness as a module (it uses package-relative imports):"
echo "  python -m harness.run_experiment -c configs/iter/<model>.yaml \\"
echo "      --mode debug --v2-task-list benchmark/task_list.json --dry-run"
