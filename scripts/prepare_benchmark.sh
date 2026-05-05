#!/usr/bin/env bash
# Materialize flat dataset layout (input/, prompts/, testbench/, references/) from
# per-instance layout (data/<id>/...) for the harness to consume.
#
# Usage:
#   bash scripts/prepare_benchmark.sh ../neurips2026_dataset
# Or, with default path inferred from sibling dir:
#   bash scripts/prepare_benchmark.sh
set -euo pipefail

DATASET_DIR="${1:-../neurips2026_dataset}"
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
[ -f "$DATASET_DIR/manifest.json" ] || { echo "ERROR: $DATASET_DIR/manifest.json not found. Provide path to neurips2026_dataset/."; exit 1; }

# The harness resolves cfg's dataset_dir ("data") against the code-package root
# (one level up from this script). Locate that root so we can also expose
# `data/` here pointing to the dataset.
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Flat layout dirs — must be siblings of the per-instance dirs, both under
# the dir the harness sees as `dataset_dir`. After the symlink below, the
# harness sees `<code_root>/data` -> `$DATASET_DIR/data`, so the flat dirs
# live INSIDE $DATASET_DIR/data alongside the per-instance dirs.
INPUT_DIR="$DATASET_DIR/data/input"
PROMPTS_DIR="$DATASET_DIR/data/prompts"
TESTBENCH_DIR="$DATASET_DIR/data/testbench"
REFERENCES_DIR="$DATASET_DIR/data/references"

mkdir -p "$INPUT_DIR" "$PROMPTS_DIR" "$TESTBENCH_DIR" "$REFERENCES_DIR"

n=0
for inst in "$DATASET_DIR"/data/*/; do
  inst_id="$(basename "$inst")"
  # Skip the bundled external-libs subtree and the flat-layout dirs themselves.
  case "$inst_id" in
    _external|input|prompts|testbench|references) continue ;;
  esac
  # Stem = inst_id minus the __error__hash suffix
  # e.g., cuda_123__logic_error__12ba71c9 -> cuda_123
  stem="${inst_id%%__*}"

  # input/<stem>.json
  ln -sf "../$inst_id/input.json" "$INPUT_DIR/$stem.json"

  # prompts/<stem>.txt
  if [ -f "$inst/prompt.txt" ]; then
    ln -sf "../$inst_id/prompt.txt" "$PROMPTS_DIR/$stem.txt"
  fi

  # testbench/<stem>/  (use -n so re-runs don't dereference the existing
  # symlink and create a nested testbench/<stem>/testbench inside the target)
  if [ -d "$inst/testbench" ]; then
    ln -sfn "../$inst_id/testbench" "$TESTBENCH_DIR/$stem"
  fi

  # references/<stem>.{cu,py} or .json
  for ref in "$inst"/reference.*; do
    [ -f "$ref" ] || continue
    ext="${ref##*.}"
    ln -sf "../$inst_id/reference.$ext" "$REFERENCES_DIR/$stem.$ext"
  done

  # External library headers (CUTLASS / ThunderKittens) used by Class 1
  # Makefile-driven testbenches. Their build_command expects
  # `./reference_sources/{cutlass,ThunderKittens}` relative to cwd. Use
  # ABSOLUTE symlink targets so the testbench can be copied to a workdir at
  # any depth without the relative target breaking. Non-Class-1 testbenches
  # simply ignore the extra dir.
  if [ -d "$DATASET_DIR/data/_external" ] && [ -d "$inst/testbench" ] \
        && [ "$inst_id" != "_external" ]; then
    mkdir -p "$inst/testbench/reference_sources"
    for ext_lib in cutlass ThunderKittens; do
      target="$DATASET_DIR/data/_external/$ext_lib"
      link="$inst/testbench/reference_sources/$ext_lib"
      if [ -d "$target" ]; then
        # Force-replace any prior link so we always end up with an absolute target.
        rm -f "$link"
        ln -s "$target" "$link"
      fi
    done
  fi

  n=$((n+1))
done

# Expose <code_root>/data -> $DATASET_DIR/data so the cfg's `dataset_dir:
# "data"` and the task_list.json's `broken_start.code_path: data/<inst>/...`
# (relative to dataset root) both resolve. The harness uses
# `Path(__file__).resolve().parents[1] / broken_start.code_path` which
# evaluates to `<code_root>/data/<inst>/...`; that needs to land in the
# dataset's per-instance directory.
# Idempotent: if a non-symlink data/ exists, leave it alone (reviewer may have
# their own layout); otherwise (re)point the symlink at the dataset's data/.
if [ -e "$CODE_ROOT/data" ] && [ ! -L "$CODE_ROOT/data" ]; then
  echo "WARNING: $CODE_ROOT/data exists and is not a symlink; not touching."
else
  ln -sfn "$DATASET_DIR/data" "$CODE_ROOT/data"
  echo "Linked $CODE_ROOT/data -> $DATASET_DIR/data"
fi

echo "Created flat layout for $n instances under:"
echo "  $INPUT_DIR/"
echo "  $PROMPTS_DIR/"
echo "  $TESTBENCH_DIR/"
echo "  $REFERENCES_DIR/"
echo
echo "Note: collisions on stem (e.g., cuda_123 appearing in both v2 and v3) will overwrite -- only one link per stem."
echo "For the 213-instance dataset, $n instances mapped to ~$(ls "$INPUT_DIR" | wc -l) unique stems."
