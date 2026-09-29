"""
Compare two checkpoint-selection rules for every unsupervised (LaSCo-trained) run:

  bench : epoch maximising mean(Fashion-IQ val avg R@10, CIRR val summary)  (current paper rule)
  lasco : epoch maximising LaSCo val R@10                                    (label-free rule)

Reads the per-epoch benchmark scores from <run_dir>/train_log.json and the LaSCo val
scores from results/lasco_val_selection/<student>/<run>/epochNN.json (produced by
scripts/eval_lasco_val_sweep.py). Ties go to the earlier epoch.

Output: results/lasco_val_selection/selection_comparison.csv
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEL_ROOT = PROJECT_ROOT / "results/lasco_val_selection"

# (student, table, family, objective, run_dir)
RUNS = [
    ("vista", "families", "reference", "contrastive FT", "results/training/vista_lasco_finetune_contrastive_v2"),
    ("vista", "families", "response", "MarginMSE", "results/bge_vl/distill_bge_vl_mse"),
    ("vista", "families", "response", "KL", "results/bge_vl/distill_bge_vl_kl"),
    ("vista", "families", "response", "ListMLE", "results/bge_vl/distill_bge_vl_listmle"),
    ("vista", "families", "response", "CE+MSE", "results/bge_vl/distill_bge_vl_ce_mse"),
    ("vista", "families", "feature", "Feature regression", "results/bge_vl/distill_bge_vl_feature"),
    ("vista", "families", "feature", "CRD-N", "results/bge_vl/distill_bge_vl_crd"),
    ("vista", "families", "relation", "SP", "results/bge_vl/distill_bge_vl_sp_w3000"),
    ("vista", "families", "relation", "RKD", "results/bge_vl/distill_bge_vl_rkd_bs128"),
    ("vista", "families", "relation", "CRD-P", "results/bge_vl/distill_bge_vl_crd_p"),
    ("vista", "teachers", "response", "CE+MSE (Qwen3-VL-Reranker-2B)", "results/training/vista_lasco_distill_ce_mse_lam05_norm"),
    ("vista", "teachers", "response", "CE+MSE (LamRA-Rank-7B)", "results/training/vista_lasco_distill_lamra_ce_mse"),
    ("vista", "teachers", "response", "CE+MSE (LamRA-Ret-7B)", "results/lamra_ret/distill_lamra_ret_ce_mse"),
    ("vista", "teachers", "relation", "SP (LamRA-Ret-7B)", "results/lamra_ret/distill_lamra_ret_sp_random_long_w3000"),
    ("magiclens", "families", "reference", "contrastive FT", "results/magiclens/magiclens_lasco_contrastive"),
    ("magiclens", "families", "response", "MarginMSE", "results/magiclens/distill_magiclens_bge_vl_mse"),
    ("magiclens", "families", "response", "KL", "results/magiclens/distill_magiclens_bge_vl_kl"),
    ("magiclens", "families", "response", "ListMLE", "results/magiclens/distill_magiclens_bge_vl_listmle"),
    ("magiclens", "families", "response", "CE+MSE", "results/magiclens/distill_magiclens_bge_vl_ce_mse"),
    ("magiclens", "families", "feature", "Feature regression", "results/magiclens/distill_magiclens_bge_vl_feature"),
    ("magiclens", "families", "feature", "CRD-N", "results/magiclens/distill_magiclens_bge_vl_crd"),
    ("magiclens", "families", "relation", "SP", "results/magiclens/distill_magiclens_bge_vl_sp_w3000_ce0.1"),
    ("magiclens", "families", "relation", "RKD", "results/magiclens/distill_magiclens_bge_vl_rkd_bs128"),
    ("magiclens", "families", "relation", "CRD-P", "results/magiclens/distill_magiclens_bge_vl_crd_p"),
]


def _bench(run_dir: Path) -> dict[int, dict]:
    out = {}
    for row in json.loads((run_dir / "train_log.json").read_text()):
        e = row.get("eval") or {}
        if "fashioniq_val_avg_recall_at10" in e and "cirr_val_summary_average" in e:
            out[row["epoch"]] = {"fiq_r10": e["fashioniq_val_avg_recall_at10"],
                                 "fiq_r50": e["fashioniq_val_avg_recall_at50"],
                                 "cirr_r1": e["cirr_val_global_recall_at1"],
                                 "cirr_rs1": e["cirr_val_subset_recall_at1"],
                                 "cirr_sum": e["cirr_val_summary_average"]}
    return out


def _lasco(student: str, run_name: str) -> dict[int, dict]:
    out = {}
    for f in sorted((SEL_ROOT / student / run_name).glob("epoch*.json")):
        d = json.loads(f.read_text())
        out[d["epoch"]] = d["metrics"]
    return out


def _argmax(scores: dict[int, float]) -> int:
    return max(sorted(scores), key=lambda e: scores[e])   # max() keeps the first (earliest) on ties


def main() -> None:
    rows = []
    for student, table, family, objective, run in RUNS:
        run_dir = PROJECT_ROOT / run
        bench, lasco = _bench(run_dir), _lasco(student, run_dir.name)
        if not lasco:
            print(f"[pending] {student} {objective}: no LaSCo val results yet")
            continue
        epochs = sorted(set(bench) & set(lasco))
        if len(epochs) != len(lasco):
            print(f"[warn] {run}: benchmark epochs {sorted(bench)} vs LaSCo epochs {sorted(lasco)}")
        eb = _argmax({e: (bench[e]["fiq_r10"] + bench[e]["cirr_sum"]) / 2 for e in epochs})
        el = _argmax({e: lasco[e]["recall_at10"] for e in epochs})
        b, l = bench[eb], bench[el]
        rows.append({
            "student": student, "table": table, "family": family, "objective": objective,
            "n_epochs": len(epochs),
            "bench_epoch": eb, "lasco_epoch": el, "same_epoch": eb == el,
            "bench_fiq_r10": round(b["fiq_r10"], 2), "lasco_fiq_r10": round(l["fiq_r10"], 2),
            "d_fiq_r10": round(l["fiq_r10"] - b["fiq_r10"], 2),
            "bench_cirr_sum": round(b["cirr_sum"], 2), "lasco_cirr_sum": round(l["cirr_sum"], 2),
            "d_cirr_sum": round(l["cirr_sum"] - b["cirr_sum"], 2),
            "bench_cirr_r1": round(b["cirr_r1"], 2), "lasco_cirr_r1": round(l["cirr_r1"], 2),
            "bench_cirr_rs1": round(b["cirr_rs1"], 2), "lasco_cirr_rs1": round(l["cirr_rs1"], 2),
            "d_cirr_rs1": round(l["cirr_rs1"] - b["cirr_rs1"], 2),
            "lasco_r10_at_bench": lasco[eb]["recall_at10"], "lasco_r10_at_lasco": lasco[el]["recall_at10"],
            "run_dir": run,
        })

    out = SEL_ROOT / "selection_comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'student':9s} {'objective':32s} ep(b/l)   FIQ R@10 b→l      CIRR sum b→l      CIRR Rs@1 b→l")
    for r in rows:
        print(f"{r['student']:9s} {r['objective']:32s} {r['bench_epoch']:2d}/{r['lasco_epoch']:<2d}   "
              f"{r['bench_fiq_r10']:5.2f}→{r['lasco_fiq_r10']:5.2f} ({r['d_fiq_r10']:+.2f})  "
              f"{r['bench_cirr_sum']:5.2f}→{r['lasco_cirr_sum']:5.2f} ({r['d_cirr_sum']:+.2f})  "
              f"{r['bench_cirr_rs1']:5.2f}→{r['lasco_cirr_rs1']:5.2f} ({r['d_cirr_rs1']:+.2f})")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
