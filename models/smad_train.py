# models/smad_train.py

"""Train a SMAD student model.

This module contains reusable training and evaluation helpers for a multi-label
classifier over {speech, music, noise} using BLOCS segment manifests.

Design:
  - train_smad(...) is the library entry point (notebook-friendly).
  - main() is a thin CLI wrapper that calls train_smad(...).

Assumes:
  - A manifest exists under Settings.metadata_path.
  - Segment audio lives under Settings.segments_path.

Example:
    python -m smad_train \
  --manifest_name data/metadata/blocs_smad_manifest_with_splits.csv \
  --model_name MIT/ast-finetuned-audioset-10-10-0.4593 \
  --batch_size 8 --num_workers 4 \
  --epochs 8 --lr 0.0003 --weight_decay 0.01 \
  --freeze_encoder \
  --use_ast_collate \
  --gold_loss_weight 3.0 --pseudo_loss_weight 1.0 \
  --train_mode all \
  --device mps
"""

from __future__ import annotations

import argparse
import json
import random
import time

from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
from datetime import datetime

import torch
from torch import nn
from torch.optim import AdamW

from config import AST_MODEL_NAME, get_settings
from utils.artifacts import V4, MANIFEST
from utils.dams_types import CLASSES


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for SMAD student training run."""
    manifest_name: str
    model_name: str

    batch_size: int = 8
    num_workers: int = 4

    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01

    freeze_encoder: bool = True
    use_ast_collate: bool = False
    use_tqdm: bool = True
    log_every_n_steps: int = 50

    # Loss weighting for mixed supervision.
    gold_loss_weight: float = 3.0
    pseudo_loss_weight: float = 1.0

    # Which examples to use in the TRAIN split only.
    # Options: 'all' (gold + pseudo), 'gold_only', 'pseudo_only'
    train_mode: str = 'all'

    threshold: float = 0.5
    select_metric: str = 'macro'  # 'micro' or 'macro'

    seed: int = 1337
    device: Optional[str] = None

    output_dir: str = 'checkpoints/smad_student'
    run_name: Optional[str] = None


def _get_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move any tensor values in batch dic onto the target device."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _split_batch(batch: Dict[str, Any]) -> tuple[Dict[str, Any], torch.Tensor]:
    """Separate model inputs from labels.

    The dataset/collate is expected to produce a 'labels' tensor in the batch.
    """
    if 'labels' not in batch:
        raise KeyError("Batch is missing 'labels'.")
    labels = batch['labels']
    model_in = {k: v for k, v in batch.items() if k != 'labels'}
    return model_in, labels


def _weighted_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    example_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCEWithLogitsLoss with optional per-example weights.

    - Computes elementwise BCE over [B, C], reduces to per-example loss via mean
      over classes, then applies optional weights over the batch.
    """
    loss_per_elem = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction='none',
    )
    # Reduce over classes to get one loss value per example.
    loss_per_ex = loss_per_elem.mean(dim=1)

    if example_weights is not None:
        w = example_weights
        if w.dim() == 2 and w.shape[1] == 1:
            w = w.squeeze(1)
        loss_per_ex = loss_per_ex * w

    return loss_per_ex.mean()


@torch.no_grad()
def _compute_prf(y_true: torch.Tensor, y_prob: torch.Tensor, thr: float) -> Dict[str, Any]:
    """Compute micro/macro and per-class precision/recall/F1 at a fixed threshold."""
    y_true = y_true.int()
    y_pred = (y_prob >= thr).int()

    eps = 1e-12
    per_class: Dict[str, Any] = {}
    micro_tp = micro_fp = micro_fn = 0

    macro_prec_sum = 0.0
    macro_rec_sum = 0.0
    macro_f1_sum = 0.0

    for i, name in enumerate(CLASSES):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        tp = int(((yp == 1) & (yt == 1)).sum().item())
        fp = int(((yp == 1) & (yt == 0)).sum().item())
        fn = int(((yp == 0) & (yt == 1)).sum().item())

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)

        per_class[name] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
        }

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        macro_prec_sum += float(prec)
        macro_rec_sum += float(rec)
        macro_f1_sum += float(f1)

    n_classes = max(len(CLASSES), 1)

    micro_prec = micro_tp / (micro_tp + micro_fp + eps)
    micro_rec = micro_tp / (micro_tp + micro_fn + eps)
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + eps)

    macro_prec = macro_prec_sum / n_classes
    macro_rec = macro_rec_sum / n_classes
    macro_f1 = macro_f1_sum / n_classes

    return {
        'micro': {
            'precision': float(micro_prec),
            'recall': float(micro_rec),
            'f1': float(micro_f1),
            'tp': int(micro_tp),
            'fp': int(micro_fp),
            'fn': int(micro_fn),
        },
        'macro': {
            'precision': float(macro_prec),
            'recall': float(macro_rec),
            'f1': float(macro_f1),
        },
        'per_class': per_class,
    }


@torch.no_grad()
def run_eval(model: nn.Module, loader, device: torch.device, thr: float) -> Dict[str, Any]:
    """Evaluate a model over a loader and compute loss + PRF metrics.

    The model is assumed to return logits. We apply sigmoid to obtain probabilities
    for thresholding.
    """

    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0

    all_true = []
    all_prob = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        model_in, labels = _split_batch(batch)

        logits = model(model_in)
        loss = loss_fn(logits, labels)

        # Apply sigmoid to logits for probability thresholding.
        probs = torch.sigmoid(logits)

        all_true.append(labels.detach().cpu())
        all_prob.append(probs.detach().cpu())

        total_loss += float(loss.item())
        n_batches += 1

    y_true = torch.cat(all_true, dim=0) if all_true else torch.zeros((0, len(CLASSES)))
    y_prob = torch.cat(all_prob, dim=0) if all_prob else torch.zeros((0, len(CLASSES)))

    metrics = _compute_prf(y_true, y_prob, thr=thr)
    metrics['loss'] = total_loss / max(n_batches, 1)
    metrics['n_examples'] = int(y_true.shape[0])

    return metrics



def _write_run_config(
        output_dir: Path,
        cfg: TrainConfig,
        device: torch.device,
        settings
) -> None:
    """Write run configuration to output directory as JSON."""

    # Persist the exact knobs and resolved paths used for this run.
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'manifest_name': cfg.manifest_name,
        'model_name': cfg.model_name,
        'batch_size': cfg.batch_size,
        'num_workers': cfg.num_workers,
        'epochs': cfg.epochs,
        'lr': cfg.lr,
        'weight_decay': cfg.weight_decay,
        'freeze_encoder': cfg.freeze_encoder,
        'use_ast_collate': cfg.use_ast_collate,
        'gold_loss_weight': cfg.gold_loss_weight,
        'pseudo_loss_weight': cfg.pseudo_loss_weight,
        'train_mode': cfg.train_mode,
        'threshold': cfg.threshold,
        'select_metric': cfg.select_metric,
        'seed': cfg.seed,
        'device': str(device),
        'metadata_path': str(settings.metadata_path),
        'segments_path': str(settings.segments_path),
    }
    with (output_dir / 'run_config.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


# --- Automatic, descriptive run naming helpers ---
# These produce a compact run directory name that encodes key hyperparameters and
# includes a short hash to avoid collisions.

def _sanitize_token(s: str) -> str:
    s = (s or '').strip()
    if not s:
        return 'na'
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch.lower())
        elif ch in ['.', '_']:
            keep.append(ch)
        else:
            keep.append('_')
    out = ''.join(keep)
    while '__' in out:
        out = out.replace('__', '_')
    return out.strip('_') or 'na'


def _model_short_name(model_name: str) -> str:
    # Example: "MIT/ast-finetuned-audioset-10-10-0.4593" -> "ast"
    mn = (model_name or '').lower()
    if 'ast' in mn:
        return 'ast'
    if 'whisper' in mn:
        return 'whisper'
    if 'clap' in mn:
        return 'clap'
    return _sanitize_token(model_name.split('/')[-1] if model_name else 'model')


def _resolve_run_name(cfg: TrainConfig) -> str:
    # Respect user-provided run_name if given.
    if cfg.run_name:
        return str(cfg.run_name)

    # Compact, high-signal encoding of config knobs.
    model_tok = _model_short_name(cfg.model_name)
    enc_tok = 'frozen' if cfg.freeze_encoder else 'unfrozen'
    mode_tok = _sanitize_token(cfg.train_mode)

    w_tok = f'g{cfg.gold_loss_weight:g}_p{cfg.pseudo_loss_weight:g}'
    lr_tok = f'lr{cfg.lr:g}'
    bs_tok = f'bs{cfg.batch_size}'
    seed_tok = f's{cfg.seed}'

    # Short, stable hash to avoid collisions.
    sig = (f'{cfg.manifest_name}|{cfg.model_name}|{enc_tok}|{mode_tok}|{w_tok}|{lr_tok}|'
           f'{bs_tok}|{seed_tok}|{cfg.epochs}|{cfg.weight_decay:g}|'
           f'{cfg.threshold:g}|{cfg.select_metric}')
    h = hashlib.sha1(sig.encode('utf-8')).hexdigest()[:8]

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return (f'{model_tok}_{enc_tok}_{mode_tok}_{w_tok}_{lr_tok}_{bs_tok}_{seed_tok}_'
            f'{h}_{ts}')


def train_smad(cfg: TrainConfig) -> Dict[str, Any]:
    """Train the SMAD student model and write artifacts to disk.

    Returns a small summary dict containing best dev score, best epoch, optional
    test metrics, and the resolved run directory.
    """
    if cfg.select_metric not in ['micro', 'macro']:
        raise ValueError("select_metric must be 'micro' or 'macro'.")

    _set_seeds(cfg.seed)

    device = _get_device(cfg.device)
    settings = get_settings()

    # Create a unique run directory so multiple experiments can coexist.
    resolved_run_name = _resolve_run_name(cfg)
    output_dir = Path(cfg.output_dir) / resolved_run_name
    _write_run_config(output_dir, cfg, device=device, settings=settings)
    with (output_dir / 'resolved_run_name.txt').open('w', encoding='utf-8') as f:
        f.write(resolved_run_name + '\n')
    # Import here to avoid circular imports.
    from models.smad_model import build_smad_model
    from models.smad_dataset import make_smad_dataloaders, build_ast_collate_fn

    collate_fn = None
    # Optionally featurize waveforms in collate_fn (often faster than inside the model).
    if cfg.use_ast_collate:
        collate_fn = build_ast_collate_fn(model_name=cfg.model_name)

    # Build dataloaders from the manifest. The loader is expected to emit 'labels'
    # and may optionally emit 'is_gold' for mixed-supervision weighting.
    train_loader, dev_loader, test_loader = make_smad_dataloaders(
        manifest_name=cfg.manifest_name,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        train_mode=cfg.train_mode,
    )

    if train_loader is None:
        raise RuntimeError('Train split is empty or missing.')

    model = build_smad_model(
        device=device,
        model_name=cfg.model_name,
        freeze_encoder=cfg.freeze_encoder,
    )

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_dev_score = -1.0
    best_epoch = None

    epoch_times_sec = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        epoch_start = time.time()

        # Optional tqdm progress bar.
        iterator = train_loader
        if cfg.use_tqdm:
            iterator = tqdm(
                train_loader,
                desc=f'Epoch {epoch}/{cfg.epochs}',
                leave=False,
                dynamic_ncols=True,
            )

        for step, batch in enumerate(iterator, start=1):
            batch = _move_batch_to_device(batch, device)
            model_in, labels = _split_batch(batch)

            optim.zero_grad(set_to_none=True)
            logits = model(model_in)

            example_weights = None
            if 'is_gold' in batch and isinstance(batch['is_gold'], torch.Tensor):
                is_gold = batch['is_gold'].to(device)
                gold_w = float(cfg.gold_loss_weight)
                pseudo_w = float(cfg.pseudo_loss_weight)
                example_weights = torch.where(
                    is_gold.bool(),
                    torch.full_like(is_gold.float(), gold_w),
                    torch.full_like(is_gold.float(), pseudo_w),
                )

            loss = _weighted_bce_with_logits(
                logits, labels, example_weights=example_weights
            )
            loss.backward()
            optim.step()

            total_loss += float(loss.item())
            n_batches += 1

            if cfg.use_tqdm:
                # Update tqdm postfix with a stable running loss.
                if step % max(int(cfg.log_every_n_steps), 1) == 0 or step == 1:
                    iterator.set_postfix({'loss': f'{(total_loss / max(n_batches, 1)):.4f}'})

        train_loss = total_loss / max(n_batches, 1)

        dev_metrics = None
        # Track the best checkpoint on the selected dev metric.
        if dev_loader is not None:
            dev_metrics = run_eval(model, dev_loader, device=device, thr=cfg.threshold)

        ckpt_latest = output_dir / 'checkpoint_latest.pt'
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'train_loss': train_loss,
                'dev_metrics': dev_metrics,
                'cfg': cfg.__dict__,
            },
            ckpt_latest,
        )

        if dev_metrics is not None:
            dev_score = float(dev_metrics[cfg.select_metric]['f1'])
            if dev_score > best_dev_score:
                best_dev_score = dev_score
                best_epoch = epoch
                ckpt_best = output_dir / 'checkpoint_best.pt'
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'train_loss': train_loss,
                        'dev_metrics': dev_metrics,
                        'best_dev_score': best_dev_score,
                        'best_dev_metric': cfg.select_metric,
                        'cfg': cfg.__dict__,
                    },
                    ckpt_best,
                )

        log_line = {
            'epoch': epoch,
            'train_loss': train_loss,
            'dev_loss': None if dev_metrics is None else dev_metrics['loss'],
            'dev_micro_f1': None if dev_metrics is None else dev_metrics['micro']['f1'],
            'dev_macro_f1': None if dev_metrics is None else dev_metrics['macro']['f1'],
            'dev_selected_metric': cfg.select_metric,
            'dev_selected_f1': None if dev_metrics is None else dev_metrics[cfg.select_metric]['f1'],
        }
        with (output_dir / 'train_log.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(log_line) + '\n')

        epoch_sec = time.time() - epoch_start
        epoch_times_sec.append(epoch_sec)

        # Simple ETA based on average epoch time so far.
        avg_epoch_sec = sum(epoch_times_sec) / max(len(epoch_times_sec), 1)
        remaining_epochs = cfg.epochs - epoch
        eta_sec = avg_epoch_sec * max(remaining_epochs, 0)

        print(
            f'Epoch {epoch} | train loss {train_loss:.4f} | epoch {epoch_sec:.1f}s |'
              f' eta {eta_sec/60.0:.1f}m'
        )
        if dev_metrics is not None:
            print(
                f'          dev loss {dev_metrics["loss"]:.4f} | '
                f'dev micro F1 {dev_metrics["micro"]["f1"]:.4f} | '
                f'dev macro F1 {dev_metrics["macro"]["f1"]:.4f} | '
                f'selected {cfg.select_metric} {dev_metrics[cfg.select_metric]["f1"]:.4f}'
            )

    test_metrics = None
    if test_loader is not None:
        ckpt_best = output_dir / 'checkpoint_best.pt'
        if ckpt_best.exists():
            best = torch.load(ckpt_best, map_location=device)
            model.load_state_dict(best['model_state_dict'])

        test_metrics = run_eval(model, test_loader, device=device, thr=cfg.threshold)
        with (output_dir / 'test_metrics.json').open('w', encoding='utf-8') as f:
            json.dump(test_metrics, f, indent=2)
        print(
            f'Test micro F1 {test_metrics["micro"]["f1"]:.4f} | '
            f'Test macro F1 {test_metrics["macro"]["f1"]:.4f} | '
            f'test loss {test_metrics["loss"]:.4f}'
        )

    return {
        'best_dev_score': float(best_dev_score),
        'best_epoch': best_epoch,
        'test': test_metrics,
        'run_dir': str(output_dir),
    }


def main() -> None:
    """CLI entry point for training."""
    p = argparse.ArgumentParser()

    p.add_argument('--manifest_name', type=str, default=None)
    p.add_argument('--model_name', type=str, default=AST_MODEL_NAME)

    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)

    p.add_argument('--freeze_encoder', action='store_true')
    p.add_argument('--unfreeze_encoder', action='store_true')

    p.add_argument('--use_ast_collate', action='store_true')

    # Progress display and logging cadence.
    p.add_argument('--no_tqdm', action='store_true')
    p.add_argument('--log_every_n_steps', type=int, default=50)

    p.add_argument('--gold_loss_weight', type=float, default=1.0)
    p.add_argument('--pseudo_loss_weight', type=float, default=1.0)

    p.add_argument(
        '--train_mode',
        type=str,
        default='all',
        choices=['all', 'gold_only', 'pseudo_only'],
        help='Training split filter: use all examples, gold only, or pseudo only.',
    )

    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--select_metric', type=str, default='macro',
                   choices=['micro', 'macro'])

    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', type=str, default=None)

    p.add_argument('--output_dir', type=str, default='checkpoints/smad_student')
    p.add_argument('--run_name', type=str, default=None)

    args = p.parse_args()

    # Resolve default manifest path
    if args.manifest_name is None:
        settings = get_settings()
        manifest_name = str(settings.manifest_dir(V4) / MANIFEST)
    else:
        manifest_name = args.manifest_name

    if args.freeze_encoder and args.unfreeze_encoder:
        raise ValueError('Choose only one of --freeze_encoder or --unfreeze_encoder.')

    freeze_encoder = True
    if args.unfreeze_encoder:
        freeze_encoder = False
    if args.freeze_encoder:
        freeze_encoder = True

    cfg = TrainConfig(
        manifest_name=manifest_name,
        model_name=args.model_name,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        freeze_encoder=bool(freeze_encoder),
        use_ast_collate=bool(args.use_ast_collate),
        use_tqdm=bool(not args.no_tqdm),
        log_every_n_steps=int(args.log_every_n_steps),
        gold_loss_weight=float(args.gold_loss_weight),
        pseudo_loss_weight=float(args.pseudo_loss_weight),
        train_mode=str(args.train_mode),
        threshold=float(args.threshold),
        select_metric=str(args.select_metric),
        seed=int(args.seed),
        device=args.device,
        output_dir=str(args.output_dir),
        run_name=args.run_name,
    )

    train_smad(cfg)


if __name__ == '__main__':
    main()
