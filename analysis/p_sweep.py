"""Exp 5: C-gate p-sweep — re-aggregates committee data at 9 p values.

Pure post-hoc analysis. Reads existing manifests + summaries from each
committee model dir; for each p value computes pass@5 per (model, task);
emits per_model_per_p table + headline dual-indicator table at p=0.3
(loose) and p=1.0 (parity).

CLI:
  python evaluation/scripts/tools/exp5_p_sweep.py \
      --committee Kimi,Qwen3.6Plus,Qwen3.6-27B,Gemma4-31B,GLM4.7 \
      --committee-dirs <comma-separated paths> \
      --task-list evaluation/v2_task_list.json \
      --out-dir evaluation/results_v2_analysis/exp5_p_sweep
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_P_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


def pass_at_5(records_for_task: list, p: float) -> bool:
    """records_for_task: ordered iter records for one (model, task).

    Asymmetric p=0.0: counts a passed-correctness iter even when speedup
    is missing (useful before Phase X.3 refill).
    """
    for r in sorted(records_for_task, key=lambda x: x.get("iteration", 0)):
        if r.get("status") != "passed":
            continue
        perf = r.get("performance") or {}
        sp = perf.get("speedup")
        if sp is None:
            if p == 0.0:
                return True
            continue
        if sp >= p:
            return True
    return False


def load_committee_data(committee_dirs: dict, task_ids: list) -> dict:
    out: dict = defaultdict(lambda: defaultdict(list))  # out[model][task]
    for label, dir_path in committee_dirs.items():
        manifest = Path(dir_path) / "index/manifest.jsonl"
        if not manifest.exists():
            print(f"WARNING: missing manifest for {label}: {manifest}")
            continue
        # Inject summary perf onto last passing iter when missing
        summary_path = Path(dir_path) / "summary.json"
        summary_perf: dict = {}
        if summary_path.exists():
            s = json.load(open(summary_path))
            for tid, info in (s.get("per_task_results") or {}).items():
                summary_perf[tid] = info.get("performance") or {}
        with open(manifest) as f:
            for line in f:
                r = json.loads(line)
                tid = r.get("task_id")
                if tid not in task_ids:
                    continue
                if "performance" not in r and r.get("status") == "passed" and tid in summary_perf:
                    r["performance"] = summary_perf[tid]
                out[label][tid].append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--committee", required=True)
    ap.add_argument("--committee-dirs", required=True)
    ap.add_argument("--task-list", default="evaluation/v2_task_list.json")
    ap.add_argument("--p-values", default=",".join(map(str, DEFAULT_P_VALUES)))
    ap.add_argument("--out-dir", default="evaluation/results_v2_analysis/exp5_p_sweep")
    args = ap.parse_args()

    labels = args.committee.split(",")
    dirs = args.committee_dirs.split(",")
    if len(labels) != len(dirs):
        ap.error("--committee labels and --committee-dirs must have same count")
    p_vals = [float(x) for x in args.p_values.split(",")]

    task_list = json.load(open(ROOT / args.task_list))
    task_ids_set = {e["task_id"] for e in task_list["entries"]}
    task_ids = list(task_ids_set)

    raw = load_committee_data(dict(zip(labels, dirs)), task_ids_set)

    per_model_per_p: dict = {}
    coverage: dict = {}
    for label in labels:
        per_model_per_p[label] = {}
        coverage[label] = sum(1 for t in task_ids if raw.get(label, {}).get(t))
        for p in p_vals:
            passes = [pass_at_5(raw.get(label, {}).get(t, []), p) for t in task_ids]
            per_model_per_p[label][f"{p:.1f}"] = mean(passes) if passes else 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / args.out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_doc = {
        "_meta": {
            "version": "v1",
            "created": datetime.now().isoformat(),
            "committee": labels,
            "p_values": p_vals,
            "data_sources": dict(zip(labels, dirs)),
            "tasks_total": len(task_ids),
            "tasks_with_records_per_model": coverage,
        },
        "per_model_per_p_pass5": per_model_per_p,
    }
    (out_dir / "p_sweep_data.json").write_text(json.dumps(out_doc, indent=2) + "\n")

    # Headline table at p=0.3 and p=1.0
    lines = [
        "## Headline pass@5 — dual indicator (p=0.3 loose, p=1.0 parity)",
        "",
        "| Model | tasks_with_records | pass@5 (p=0.3) | pass@5 (p=1.0) | Δ |",
        "|---|---|---|---|---|",
    ]
    for m in labels:
        p3 = per_model_per_p[m].get("0.3", 0.0)
        p1 = per_model_per_p[m].get("1.0", 0.0)
        lines.append(
            f"| {m} | {coverage.get(m, 0)} | {p3:.1%} | {p1:.1%} | {p1 - p3:+.1%} |"
        )
    (out_dir / "main_table.md").write_text("\n".join(lines) + "\n")

    # Full per-p table
    lines2 = ["## Full p-sweep — pass@5 per model per p", "",
              "| Model | " + " | ".join(f"p={p}" for p in p_vals) + " |",
              "|---|" + "---|" * len(p_vals)]
    for m in labels:
        row = [m] + [f"{per_model_per_p[m].get(f'{p:.1f}', 0.0):.1%}" for p in p_vals]
        lines2.append("| " + " | ".join(row) + " |")
    (out_dir / "p_sweep_full_table.md").write_text("\n".join(lines2) + "\n")

    print(f"Wrote {out_dir}")
    print(f"Coverage: {coverage}")


if __name__ == "__main__":
    main()
