# Data

No images or annotations are redistributed here. Each dataset is obtained from its own
source and placed under `data/` with the layout below; the paths in `configs/` assume it.

| Dataset | Used for | Layout expected |
|---|---|---|
| **LaSCo** | training (a fixed random subset of 40,000 of its 359,268 triplets) | `data/lasco/lasco_train.json`, `lasco_val.json`, `lasco_val_corpus.json`, `images/` (COCO train2014 and val2014) |
| **Fashion-IQ** | evaluation, validation split | `data/fashioniq/captions/`, `image_splits/`, `images/` |
| **CIRR** | evaluation, validation split and test server | `data/cirr/captions/`, `image_splits/`, `images/` |
| **CIRCO** | evaluation, test server (COCO unlabeled2017 gallery) | `data/circo/annotations/`, `images/` |

LaSCo and CIRCO reuse COCO images, so the COCO downloads can be shared between them.

Derived files, produced by the precompute scripts rather than downloaded:

| File | Produced by | Used by |
|---|---|---|
| `data/lasco/distill_subset.json` | the 40k subset draw | every training run |
| `data/lasco/teacher_scores_*.json` | `scripts/precompute_*_scores*.py` | response-based objectives |
| `data/lasco/*_cand_emb.pt` | `scripts/precompute_bge_vl_cand_emb.py` | feature- and relation-based objectives |
| `data/lasco/relation_groups_*.json` | `scripts/build_relation_groups.py` | relation-based objectives |

Model weights (the students' initial checkpoints and the teachers) go under `models/` and
come from their original releases; none are included here.
