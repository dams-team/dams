from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from finetune.ast_model import ASTClassifier, get_feature_extractor

CLASSES = ["speech", "music", "noise"]


class GoldEvalDataset(Dataset):
    def __init__(self, gold_df: pd.DataFrame, segments_dir: Path, sample_rate: int = 16000):
        self.df = gold_df.reset_index(drop=True)
        self.segments_dir = Path(segments_dir)
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        wav_path = self.segments_dir / row["segment_path"]
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        labels = torch.tensor([row[f"{c}_gold"] for c in CLASSES], dtype=torch.float32)
        return {"waveform": waveform, "labels": labels}


def collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    wavs = [b["waveform"].squeeze(0) for b in batch]  # list of (T,)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return wavs, labels


def evaluate(args: argparse.Namespace) -> None:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    processor = get_feature_extractor(args.ast_model)

    manifest = pd.read_csv(args.manifest)
    gold = pd.read_csv(args.gold)
    if args.gold_filter == "irr":
        gold = gold[gold["is_irr_segment"]]
    elif args.gold_filter == "non-irr":
        gold = gold[~gold["is_irr_segment"]]

    # Keep only gold rows present in manifest
    gold = gold.merge(manifest[["segment_path"]], on="segment_path", how="inner")
    if gold.empty:
        raise ValueError("No overlapping segment_path between gold and manifest for the selected filter.")

    ds = GoldEvalDataset(gold, args.segments_dir, sample_rate=args.sample_rate)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Load model checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ASTClassifier(model_name=args.ast_model, num_labels=len(CLASSES))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_true, all_prob = [], []
    with torch.no_grad():
        for wavs, labels in loader:
            inputs = processor(wavs, sampling_rate=processor.sampling_rate, return_tensors="pt", padding=True)
            input_values = inputs["input_values"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            logits = model(input_values=input_values, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            all_true.append(labels)
            all_prob.append(probs.cpu())

    y_true = torch.cat(all_true).numpy()
    y_prob = torch.cat(all_prob).numpy()
    y_pred = (y_prob >= args.threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    metrics = {c: {"precision": float(p), "recall": float(r), "f1": float(f)} for c, p, r, f in zip(CLASSES, prec, rec, f1)}
    macro_f1 = float(f1.mean())
    print(f"Gold filter={args.gold_filter} threshold={args.threshold}")
    print(f"Per-class metrics: {metrics}")
    print(f"Macro F1: {macro_f1:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AST student on gold labels.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/blocs_smad_v2_finetune.csv"))
    parser.add_argument("--segments-dir", type=Path, default=Path("data/segments"))
    parser.add_argument("--gold", type=Path, default=Path("data/metadata/blocs_smad_gold_annotations_v1.csv"))
    parser.add_argument("--gold-filter", choices=["irr", "non-irr", "all"], default="irr")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ast-model", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
