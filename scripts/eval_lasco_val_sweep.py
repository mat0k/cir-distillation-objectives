"""
Evaluate every epoch checkpoint of one or more training runs on the LaSCo val split,
for label-free checkpoint selection (no Fashion-IQ / CIRR annotation involved).

Same protocol as scripts/eval_lasco_val.py (30,037 queries, 39,826 COCO val2014
gallery images, Recall@K), but much faster for sweeps: the encoders that no run
trains are computed once and shared, and only the trained weights are swapped in
per checkpoint.

  VISTA     : checkpoints differ from the base only in visual_proj and the top
              BGE layers; the frozen EVA-CLIP patch tokens are computed once per
              batch and reused for every checkpoint.
  MagicLens : checkpoints differ only in the fusion stack; CLIP image/text
              embeddings are computed once, then each checkpoint only runs _fuse.

Which weights to swap is found by diffing each checkpoint against the base
weights, and the script refuses to run if a checkpoint changed a frozen encoder.

Results: <out_root>/<student>/<run_name>/epochNN.json  (zero-shot: zero_shot.json).
Existing results are skipped, so a job can be resubmitted after a failure.

Usage
-----
python scripts/eval_lasco_val_sweep.py --student vista --zero_shot \\
    --runs results/bge_vl/distill_bge_vl_kl results/bge_vl/distill_bge_vl_sp_w3000
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.lasco_eval import _CorpusDataset, _QueryDataset
from src.retrievers.vista_retriever import VistaImageProcessor

VISTA_BASE = "models/Visualized_BGE/Visualized_base_en_v1.5.pth"
MAGICLENS_BASE = "models/magiclens/magic_lens_clip_base.pt"
# Weights no run may change; the shared features are computed from them.
FROZEN_PREFIX = {"vista": "model_visual.", "magiclens": "clip."}

IMAGES_DIR = PROJECT_ROOT / "data/lasco/images"
VAL_PATH = PROJECT_ROOT / "data/lasco/lasco_val.json"
CORPUS_PATH = PROJECT_ROOT / "data/lasco/lasco_val_corpus.json"

logger = logging.getLogger(__name__)


@dataclass
class Target:
    run_name: str
    epoch: int            # 0 = zero-shot base weights
    checkpoint: str
    out_path: Path
    diff: dict = field(default_factory=dict)   # key -> tensor on GPU
    query: list = field(default_factory=list)
    gallery: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Targets and weight swapping
# ---------------------------------------------------------------------------

def _collect_targets(args) -> list[Target]:
    out_dir = PROJECT_ROOT / args.out_root / args.student
    targets = []
    if args.zero_shot:
        base = VISTA_BASE if args.student == "vista" else MAGICLENS_BASE
        targets.append(Target("zero_shot", 0, base, out_dir / "zero_shot.json"))
    for run in args.runs:
        run_dir = PROJECT_ROOT / run
        ckpts = sorted((run_dir / "checkpoints").glob(f"{args.student}_epoch*.pth"))
        if not ckpts:
            raise FileNotFoundError(f"no {args.student}_epochNN.pth in {run_dir}/checkpoints")
        for ckpt in ckpts:
            epoch = int(re.search(r"epoch(\d+)", ckpt.name).group(1))
            targets.append(Target(run_dir.name, epoch, str(ckpt.relative_to(PROJECT_ROOT)),
                                  out_dir / run_dir.name / f"epoch{epoch:02d}.json"))
    todo = [t for t in targets if not t.out_path.exists()]
    logger.info(f"{len(targets)} checkpoints, {len(targets) - len(todo)} already done, {len(todo)} to run")
    return todo


def _load_diffs(targets: list[Target], ref_cpu: dict, student: str, device) -> None:
    frozen = FROZEN_PREFIX[student]
    for t in targets:
        if t.epoch == 0:
            continue
        state = torch.load(str(PROJECT_ROOT / t.checkpoint), map_location="cpu")
        if set(state) != set(ref_cpu):
            raise KeyError(f"{t.checkpoint}: keys differ from the base model")
        t.diff = {k: v.to(device) for k, v in state.items() if not torch.equal(v, ref_cpu[k])}
        changed_frozen = [k for k in t.diff if k.startswith(frozen)]
        if changed_frozen:
            raise RuntimeError(f"{t.checkpoint} changes frozen weights ({changed_frozen[:3]}...); "
                               "shared features would be wrong for it")
        logger.info(f"  {t.run_name} ep{t.epoch:02d}: {len(t.diff)} trained tensors")
        del state


class Swapper:
    """Loads a checkpoint's trained tensors into the live model and restores the base."""

    def __init__(self, model: torch.nn.Module, targets: list[Target]):
        self.live = model.state_dict()   # shares storage with the parameters
        keys = set().union(*(t.diff for t in targets)) if targets else set()
        self.base = {k: self.live[k].clone() for k in keys}

    @torch.no_grad()
    def apply(self, t: Target) -> None:
        for k, v in t.diff.items():
            self.live[k].copy_(v)

    @torch.no_grad()
    def restore(self, t: Target) -> None:
        for k in t.diff:
            self.live[k].copy_(self.base[k])


# ---------------------------------------------------------------------------
# Embedding passes
# ---------------------------------------------------------------------------

def _loaders(image_tf, caption_tf, args):
    corpus = _CorpusDataset(str(CORPUS_PATH), IMAGES_DIR, image_tf)
    queries = _QueryDataset(str(VAL_PATH), IMAGES_DIR, image_tf, caption_tf)
    kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return DataLoader(corpus, **kw), DataLoader(queries, **kw)


@torch.no_grad()
def _embed_vista(model, targets, swapper, args):
    device = model.device
    corpus_loader, query_loader = _loaders(
        VistaImageProcessor(model.preprocess_val), model.tokenizer, args)
    patch_tokens = model.img_token_embedding          # bound method, frozen EVA-CLIP
    model.img_token_embedding = lambda tokens: tokens  # encode_mm now takes cached tokens

    gallery_ids, target_ids = [], []
    for batch in tqdm(corpus_loader, desc="Gallery"):
        tokens = patch_tokens(batch["image"].to(device))
        for t in targets:
            swapper.apply(t)
            t.gallery.append(model.encode_image(tokens).cpu())
            swapper.restore(t)
        gallery_ids.append(batch["image_id"])

    for batch in tqdm(query_loader, desc="Queries"):
        tokens = patch_tokens(batch["ref_image"].to(device))
        texts = {"input_ids": batch["input_ids"].to(device),
                 "attention_mask": batch["attention_mask"].to(device)}
        for t in targets:
            swapper.apply(t)
            t.query.append(model.encode_mm(tokens, texts).cpu())
            swapper.restore(t)
        target_ids.append(batch["target_image_id"])

    del model.img_token_embedding
    return torch.cat(gallery_ids), torch.cat(target_ids)


@torch.no_grad()
def _embed_magiclens(model, targets, swapper, args):
    device = model.device
    corpus_loader, query_loader = _loaders(
        VistaImageProcessor(model.preprocess_val), model.tokenizer, args)

    # Frozen CLIP features, computed once with the base weights.
    g_img, gallery_ids, q_img, q_txt, target_ids = [], [], [], [], []
    for batch in tqdm(corpus_loader, desc="Gallery CLIP"):
        g_img.append(model.clip.encode_image(batch["image"].to(device)))
        gallery_ids.append(batch["image_id"])
    for batch in tqdm(query_loader, desc="Queries CLIP"):
        q_img.append(model.clip.encode_image(batch["ref_image"].to(device)))
        q_txt.append(model.clip.encode_text(batch["input_ids"].to(device)))
        target_ids.append(batch["target_image_id"])
    g_img, q_img, q_txt = torch.cat(g_img), torch.cat(q_img), torch.cat(q_txt)
    null_txt = model.clip.encode_text(model._null_text_tokens.to(device))   # [1, D]

    chunk = 4096
    for t in tqdm(targets, desc="Fusion per checkpoint"):
        swapper.apply(t)
        for s in range(0, len(g_img), chunk):
            img = g_img[s:s + chunk]
            t.gallery.append(model._fuse(img, null_txt.expand(len(img), -1)).cpu())
        for s in range(0, len(q_img), chunk):
            t.query.append(model._fuse(q_img[s:s + chunk], q_txt[s:s + chunk]).cpu())
        swapper.restore(t)
    return torch.cat(gallery_ids), torch.cat(target_ids)


# ---------------------------------------------------------------------------
# Metrics and output
# ---------------------------------------------------------------------------

@torch.no_grad()
def _recall(query, gallery, gallery_ids, target_ids, k_values, device) -> dict:
    gallery = gallery.to(device)
    gallery_ids = gallery_ids.to(device)
    hits = torch.zeros(len(k_values), dtype=torch.long)
    for s in range(0, len(query), 1024):
        sims = query[s:s + 1024].to(device) @ gallery.T
        ranked = gallery_ids[sims.topk(max(k_values), dim=1).indices]
        match = ranked == target_ids[s:s + 1024].to(device).unsqueeze(1)
        for i, k in enumerate(k_values):
            hits[i] += match[:, :k].any(dim=1).sum().cpu()
    return {f"recall_at{k}": round(100.0 * hits[i].item() / len(query), 4)
            for i, k in enumerate(k_values)}


def _save(t: Target, metrics: dict, args) -> None:
    t.out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "lasco_val",
        "student": args.student,
        "run_name": t.run_name,
        "epoch": t.epoch,
        "checkpoint": t.checkpoint,
        "eval_settings": {"num_queries": 30037, "gallery_size": 39826,
                          "tf32": not args.no_tf32, "gpu": torch.cuda.get_device_name(0)},
        "metrics": metrics,
    }
    with open(t.out_path, "w") as f:
        json.dump(result, f, indent=2)


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("transformers").setLevel(logging.WARNING)
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    torch.backends.cudnn.allow_tf32 = not args.no_tf32

    targets = _collect_targets(args)
    if not targets:
        return

    if args.student == "vista":
        from src.retrievers.backbones.vista.modeling import Visualized_BGE
        model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                               model_weight=str(PROJECT_ROOT / VISTA_BASE),
                               negatives_cross_device=False, from_pretrained=None).cuda()
    else:
        from src.retrievers.magiclens_retriever import MagicLensRetriever
        model = MagicLensRetriever.from_pretrained(
            "base", checkpoint_path=str(PROJECT_ROOT / MAGICLENS_BASE)).backbone.cuda()
    model.eval()
    device = model.device

    ref_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    _load_diffs(targets, ref_cpu, args.student, device)
    del ref_cpu
    swapper = Swapper(model, targets)

    t0 = time.time()
    embed = _embed_vista if args.student == "vista" else _embed_magiclens
    gallery_ids, target_ids = embed(model, targets, swapper, args)
    logger.info(f"Embeddings for {len(targets)} checkpoints in {(time.time() - t0) / 60:.1f} min")

    k_values = sorted(args.k)
    for t in targets:
        metrics = _recall(torch.cat(t.query), torch.cat(t.gallery), gallery_ids, target_ids,
                          k_values, device)
        _save(t, metrics, args)
        logger.info(f"{t.run_name:45s} ep{t.epoch:02d}  " +
                    "  ".join(f"R@{k}={metrics[f'recall_at{k}']:.2f}" for k in k_values))
        t.query, t.gallery = [], []


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LaSCo val sweep over all epoch checkpoints.")
    p.add_argument("--student", choices=["vista", "magiclens"], required=True)
    p.add_argument("--runs", nargs="*", default=[], help="Run dirs (relative to repo) with checkpoints/.")
    p.add_argument("--zero_shot", action="store_true", help="Also evaluate the base weights.")
    p.add_argument("--out_root", default="results/lasco_val_selection")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--no_tf32", action="store_true", help="Exact fp32 matmuls (slower).")
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    main(p.parse_args())
