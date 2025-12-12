from __future__ import annotations

import argparse
from pathlib import Path

from scripts.train_student import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AST student finetuning with recommended defaults.")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/blocs_smad_v2_finetune.csv"))
    parser.add_argument("--segments-dir", type=Path, default=Path("data/segments"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size-ast", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--ast-model", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/student_ast.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Build a namespace compatible with train()
    train_args = argparse.Namespace(
        manifest=args.manifest,
        segments_dir=args.segments_dir,
        sample_rate=16000,
        n_mels=128,
        hop_length=160,
        win_length=400,
        batch_size_ast=args.batch_size_ast,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        output=args.output,
        ast_model=args.ast_model,
    )
    train(train_args)


if __name__ == "__main__":
    main()
