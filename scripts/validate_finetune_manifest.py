from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

METADATA_DIR = Path("data/metadata")
MANIFEST = METADATA_DIR / "blocs_smad_v2_finetune.csv"
GOLD = METADATA_DIR / "blocs_smad_gold_annotations_v1.csv"

REQUIRED_TEACHER_COLS = [
    # AST
    "ast_speech_score",
    "ast_music_score",
    "ast_noise_score",
    "ast_speech_label",
    "ast_music_label",
    "ast_noise_label",
    # Whisper
    "whisper_speech_score",
    "whisper_music_score",
    "whisper_noise_score",
    "whisper_speech_label",
    "whisper_music_label",
    "whisper_noise_label",
    # CLAP
    "clap_speech_score",
    "clap_music_score",
    "clap_noise_score",
    "clap_speech_label",
    "clap_music_label",
    "clap_noise_label",
    # M2D
    "m2d_speech_score",
    "m2d_music_score",
    "m2d_noise_score",
    "m2d_speech_label",
    "m2d_music_label",
    "m2d_noise_label",
]

CHOSEN_COLS = [
    "chosen_speech_label",
    "chosen_music_label",
    "chosen_noise_label",
    "chosen_speech_score",
    "chosen_music_score",
    "chosen_noise_score",
]


def main() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    print(f"Loaded manifest: {MANIFEST} rows={len(df)} columns={df.shape[1]}")

    dup = df["segment_path"].duplicated().sum()
    if dup:
        raise ValueError(f"Found {dup} duplicate segment_path entries")
    print("No duplicate segment_path entries.")

    missing_cols = [c for c in REQUIRED_TEACHER_COLS + CHOSEN_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    print("All required teacher and chosen columns present.")

    nulls = df[CHOSEN_COLS].isnull().sum()
    if nulls.any():
        raise ValueError(f"Nulls in chosen columns:\n{nulls}")
    print("Chosen columns have no nulls.")

    for col in ["chosen_speech_label", "chosen_music_label", "chosen_noise_label"]:
        vc = df[col].value_counts().to_dict()
        print(f"Value counts for {col}: {vc}")

    if GOLD.exists():
        gold = pd.read_csv(GOLD)
        irr = gold[gold["is_irr_segment"]]
        if irr.empty:
            print("No IRR subset found in gold; skipping gold sanity.")
        else:
            merged = irr.merge(df, on="segment_path", how="inner")
            print(f"Merged IRR gold rows: {len(merged)}")
            for label in ["speech", "music", "noise"]:
                y_true = merged[f"{label}_gold"]
                y_pred = merged[f"chosen_{label}_label"]
                print(f"\nIRR gold sanity for {label}:")
                print(classification_report(y_true, y_pred, zero_division=0, digits=4))
    else:
        print(f"Gold file not found at {GOLD}; skipping gold sanity.")


if __name__ == "__main__":
    main()
