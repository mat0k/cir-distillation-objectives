"""
Build the paper's ablation table (Table 5) from the training logs: seeds and
contrastive weight, for SP distillation from BGE-VL-MLLM-S1.

The group-composition runs are kept here too, although the paper reports them only
as a sentence: they use the LamRA-Ret teacher and VISTA alone, so they do not
belong in a table whose other rows share one teacher and both students.

Every run is scored at the epoch selected by the paper rule (the one maximising the
mean of Fashion-IQ val avg R@10 and the CIRR val summary) and reported with the
table's metrics: Fashion-IQ R@10, CIRR R@1 and CIRR R$_s$@1. Runs that have not
finished yet are listed as pending.

Output: results/paper_results/ablations_table.csv
"""

import csv
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results/paper_results/ablations_table.csv"

SEEDS = {
    "VISTA": ["results/bge_vl/distill_bge_vl_sp_w3000",
              "results/bge_vl/distill_bge_vl_sp_w3000_seed43",
              "results/bge_vl/distill_bge_vl_sp_w3000_seed44"],
    "MagicLens-B": ["results/magiclens/distill_magiclens_bge_vl_sp_w3000_ce0.1",
                    "results/magiclens/distill_magiclens_bge_vl_sp_w3000_ce0.1_seed43",
                    "results/magiclens/distill_magiclens_bge_vl_sp_w3000_ce0.1_seed44"],
}
LAMBDAS = [
    ("VISTA", 1.0, "results/bge_vl/distill_bge_vl_sp_w3000"),
    ("VISTA", 0.3, "results/bge_vl/distill_bge_vl_sp_w3000_ce0.3"),
    ("VISTA", 0.1, "results/bge_vl/distill_bge_vl_sp_w3000_ce0.1"),
    ("VISTA", 0.03, "results/bge_vl/distill_bge_vl_sp_w3000_ce0.03"),
    ("MagicLens-B", 1.0, "results/magiclens/distill_magiclens_bge_vl_sp_w3000"),
    ("MagicLens-B", 0.3, "results/magiclens/distill_magiclens_bge_vl_sp_w3000_ce0.3"),
    ("MagicLens-B", 0.1, "results/magiclens/distill_magiclens_bge_vl_sp_w3000_ce0.1"),
    ("MagicLens-B", 0.03, "results/magiclens/distill_magiclens_bge_vl_sp_w3000_ce0.03"),
]
# Group composition: same teacher (LamRA-Ret), sp_weight and budget; only the
# grouping of the 128 candidates differs. Reported with the teacher's own measure
# of how many in-group candidates are valid matches for the query.
GROUPS = [
    ("random", "results/lamra_ret/distill_lamra_ret_sp_random"),
    ("moderate (k=32)", "results/lamra_ret/distill_lamra_ret_sp_moderate"),
    ("tight (k=200)", "results/lamra_ret/distill_lamra_ret_sp_tight"),
]


def _selected(run: str):
    log = PROJECT_ROOT / run / "train_log.json"
    if not log.exists():
        return None
    epochs = [r for r in json.loads(log.read_text()) if r.get("eval")]
    if not epochs:
        return None
    best = max(epochs, key=lambda r: (r["eval"]["fashioniq_val_avg_recall_at10"]
                                      + r["eval"]["cirr_val_summary_average"]) / 2)
    e = best["eval"]
    return {"epoch": best["epoch"],
            "fiq_r10": e["fashioniq_val_avg_recall_at10"],
            "cirr_r1": e["cirr_val_global_recall_at1"],
            "cirr_rs1": e["cirr_val_subset_recall_at1"],
            "false_neg_frac": best["train"].get("avg_false_neg_frac")}


def _row(block, setting, student, run, sel):
    return {"block": block, "setting": setting, "student": student,
            "epoch": sel["epoch"],
            "fiq_r10": round(sel["fiq_r10"], 2),
            "cirr_r1": round(sel["cirr_r1"], 2),
            "cirr_rs1": round(sel["cirr_rs1"], 2),
            # Only the group rows report it; it is what those rows are about.
            "false_neg_pct": (round(100 * sel["false_neg_frac"], 2)
                              if block == "groups" and sel["false_neg_frac"] is not None else ""),
            "run_dir": run}


def main() -> None:
    rows, pending = [], []

    for student, runs in SEEDS.items():
        sels = [(r, _selected(r)) for r in runs]
        done = [(r, s) for r, s in sels if s]
        pending += [r for r, s in sels if not s]
        if len(done) < 2:
            continue
        agg = {}
        for key in ("fiq_r10", "cirr_r1", "cirr_rs1"):
            vals = [s[key] for _, s in done]
            agg[key] = f"{statistics.mean(vals):.2f}+-{statistics.stdev(vals):.2f}"
        rows.append({"block": "seeds", "setting": f"SP, {len(done)} runs", "student": student,
                     "epoch": "/".join(str(s["epoch"]) for _, s in done),
                     "fiq_r10": agg["fiq_r10"], "cirr_r1": agg["cirr_r1"],
                     "cirr_rs1": agg["cirr_rs1"], "false_neg_pct": "",
                     "run_dir": ";".join(r for r, _ in done)})

    for student, lam, run in LAMBDAS:
        sel = _selected(run)
        (rows.append(_row("lambda_con", f"lambda_con={lam:g}", student, run, sel))
         if sel else pending.append(run))

    for label, run in GROUPS:
        sel = _selected(run)
        (rows.append(_row("groups", f"groups: {label}", "VISTA", run, sel))
         if sel else pending.append(run))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'block':11s} {'setting':18s} {'student':12s} {'FIQ R@10':>12s} {'CIRR R@1':>12s} {'R_s@1':>12s}  {'FN%':>5s}")
    for r in rows:
        print(f"{r['block']:11s} {r['setting']:18s} {r['student']:12s} {str(r['fiq_r10']):>12s} "
              f"{str(r['cirr_r1']):>12s} {str(r['cirr_rs1']):>12s}  {str(r['false_neg_pct']):>5s}")
    for p in pending:
        print(f"[pending] {p}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
