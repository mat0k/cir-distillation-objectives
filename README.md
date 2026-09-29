# Structure over Scores: Comparing Distillation Objectives for Composed Image Retrieval

Code for the paper. Everything here trains or evaluates a **compact composed image
retrieval (CIR) student** that is distilled from a large teacher **without any labels from
the evaluation benchmarks**: training uses LaSCo only, and the teacher supervision is
computed offline.

Two students, nine distillation objectives across three families, four teachers.

| | |
|---|---|
| Students | VISTA (EVA-CLIP-B/16 + BGE-base, 196M) and MagicLens-B (CLIP-B/16 + fusion stack, 166M) |
| Main teacher | BGE-VL-MLLM-S1 (7.6B), trained only on synthetic MegaPairs |
| Other teachers (RQ3 only) | LamRA-Ret-7B, LamRA-Rank-7B, Qwen3-VL-Reranker-2B |
| Training corpus | LaSCo, a fixed random subset of 40,000 triplets |
| Benchmarks | Fashion-IQ (val), CIRR (test server), CIRCO (test server) |

## Layout

```
src/retrievers/        students: VISTA and the MagicLens PyTorch port
src/rerankers/         reranker teachers (Qwen3-VL, LamRA-Rank)
src/datasets/          LaSCo training sets, one per supervision type
src/training/          trainer with all nine objectives + projection heads
src/evaluation/        Fashion-IQ, CIRR, CIRCO, LaSCo-val evaluation
scripts/               entry points (see below)
configs/training/      one file per run reported in the paper
configs/distillation/  teacher-supervision precomputation
results/               the CSVs behind the tables in the paper
```

## Distillation objectives

Implemented in `src/training/trainer.py`, selected by `distillation.loss` in a config:

| Family | Objectives | What the student matches |
|---|---|---|
| Response | MarginMSE, KL, ListMLE, CE+MSE | the teacher's scores over a pool of candidates |
| Feature | feature regression, CRD-N | each item's position in the teacher's space |
| Relation | SP, RKD, CRD-P | the arrangement of items in that space |

## Reproducing a run

1. **Get the data.** See [`data/README.md`](data/README.md) for LaSCo, Fashion-IQ, CIRR and CIRCO.
2. **Precompute the teacher's supervision** (once per teacher; no teacher runs during training):

   ```bash
   python scripts/precompute_bge_vl_scores.py     # scores over candidate pools
   python scripts/precompute_bge_vl_cand_emb.py   # candidate embeddings (feature/relation)
   python scripts/build_relation_groups.py        # groups for relation objectives
   ```

3. **Train**, one config per row of the paper's tables:

   ```bash
   python scripts/train_retriever.py --config configs/training/lasco_distill_bge_vl_sp_w3000.yaml
   python scripts/train_retriever.py --config configs/training/lasco_distill_magiclens_bge_vl_sp_ce0.1.yaml
   ```

   Seeds are overridden on the command line: `--seed 43 --run_name <name>`.
   Each run writes per-epoch Fashion-IQ and CIRR validation scores to
   `<output_dir>/train_log.json`, which is what checkpoint selection reads.

4. **Evaluate.** Validation scores come from the training log; the test splits go through the
   official servers:

   ```bash
   python scripts/generate_cirr_test_submission.py --only <run>
   python scripts/generate_circo_test_submission.py --only <run>
   python scripts/run_baseline.py --config <config>     # zero-shot / baseline numbers
   ```

5. **Rebuild the tables:**

   ```bash
   python scripts/summarize_ablations.py        # ablation table (seeds, contrastive weight)
   python scripts/compare_lasco_selection.py    # checkpoint-selection comparison
   ```

## Checkpoint selection

One epoch per run is selected on the validation splits, the one maximising the mean of
Fashion-IQ R@10 and the CIRR summary score (the mean of R@5 and R_s@1). No test split is
used for any decision, and CIRCO is used for neither training nor selection.
`scripts/eval_lasco_val_sweep.py` and `scripts/compare_lasco_selection.py` reproduce the
comparison against selecting on a held-out split of the training corpus instead.

## MagicLens port

MagicLens is released only in JAX. `scripts/convert_magiclens_weights.py` converts the
released checkpoint to PyTorch and `scripts/check_magiclens_parity.py` verifies that the
embeddings match the original to a cosine similarity of 1.000.

## Efficiency

Distillation changes only the weights, so a distilled student has exactly the cost of its
backbone. `scripts/benchmark_efficiency.py` measures parameters, GFLOPs, latency and index
size for both arms; `scripts/benchmark_teacher_flops.py` does the same for the teacher,
which runs only during the offline precomputation.

## Requirements

`pip install -r requirements.txt`, one GPU. Training a student takes a few hours on one
A100 partition; the teacher precomputation is the only step needing a large GPU.
