from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from finetune.dataset import SmadDataset, CLASSES
from finetune.ast_model import ASTClassifier, get_feature_extractor


def compute_pos_weight(dataset: SmadDataset) -> torch.Tensor:
    labels = np.stack([dataset.df[f"chosen_{c}_label"].to_numpy() for c in CLASSES], axis=1)
    pos_counts = labels.sum(axis=0)
    neg_counts = labels.shape[0] - pos_counts
    # Avoid divide by zero
    pos_weight = np.where(pos_counts == 0, 1.0, neg_counts / (pos_counts + 1e-6))
    return torch.tensor(pos_weight, dtype=torch.float32)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, processor=None) -> Tuple[dict, float]:
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            labels = batch["labels"].to(device)
            # Feature extractor expects a list of 1D waveforms, not a 2D batch tensor.
            wavs = batch["waveform"].squeeze(1)  # (B, T)
            wav_list = [w.cpu().numpy() for w in wavs]  # list of (T,)
            inputs = processor(wav_list, sampling_rate=processor.sampling_rate, return_tensors="pt", padding=True)
            input_values = inputs["input_values"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            logits = model(input_values=input_values, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    metrics = {c: {"precision": float(p), "recall": float(r), "f1": float(f)} for c, p, r, f in zip(CLASSES, prec, rec, f1)}
    macro_f1 = float(f1.mean())
    return metrics, macro_f1


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = get_feature_extractor(args.ast_model)
    full_ds = SmadDataset(
        manifest_path=args.manifest,
        segments_dir=args.segments_dir,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        win_length=args.win_length,
        return_waveform=True,
    )

    val_size = int(len(full_ds) * args.val_fraction)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    pos_weight = compute_pos_weight(full_ds).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = ASTClassifier(model_name=args.ast_model, num_labels=len(CLASSES)).to(device)
    # AST has its own feature extractor padding; use smaller batch sizes to fit memory
    train_loader = DataLoader(train_ds, batch_size=args.batch_size_ast, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_ast, shuffle=False, num_workers=2, pin_memory=True)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False)
        for batch in train_pbar:
            labels = batch["labels"].to(device)
            # Feature extractor expects a list of 1D waveforms, not a 2D batch tensor.
            wavs = batch["waveform"].squeeze(1)  # (B, T)
            wav_list = [w.cpu().numpy() for w in wavs]  # list of (T,)
            inputs = processor(wav_list, sampling_rate=processor.sampling_rate, return_tensors="pt", padding=True)
            input_values = inputs["input_values"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            logits = model(input_values=input_values, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item() * input_values.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss = running_loss / len(train_loader.dataset)
        val_metrics, macro_f1 = evaluate(model, val_loader, device, processor=processor)
        print(f"Epoch {epoch}: loss={epoch_loss:.4f} val_macro_f1={macro_f1:.4f} val_metrics={val_metrics}")

    # Optional: save final weights
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, out_path)
        print(f"Saved model to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple SMAD student on finetune manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/blocs_smad_v2_finetune.csv"))
    parser.add_argument("--segments-dir", type=Path, default=Path("data/segments"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--batch-size-ast", type=int, default=2, help="Batch size for AST backbone (memory heavier).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/student_ast.pt"))
    parser.add_argument("--ast-model", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
