from __future__ import annotations

import argparse
import json
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from datasets import Dataset, load_from_disk


# Mapping: teacher name -> (dataset folder name, columns to extract)
TEACHER_SPECS: Dict[str, Tuple[str, List[str]]] = {
    "ast": (
        "blocs_smad_v2_ast",
        ["speech_score", "music_score", "noise_score", "speech_label", "music_label", "noise_label", "ast_probs"],
    ),
    "clap": (
        "blocs_smad_v2_clap",
        ["speech_score", "music_score", "noise_score", "speech_label", "music_label", "noise_label"],
    ),
    "m2d": (
        "blocs_smad_v2_m2d",
        ["speech_score", "music_score", "noise_score", "speech_label", "music_label", "noise_label", "m2d_clap_scores"],
    ),
    "whisper": (
        "blocs_smad_v2_whisper",
        ["speech_score", "music_score", "noise_score", "speech_label", "music_label", "noise_label", "whisper_probs"],
    ),
}


def load_teacher_df(metadata_dir: Path, name: str, folder: str, cols: List[str]) -> pd.DataFrame:
    ds_path = metadata_dir / folder
    df = load_from_disk(ds_path).to_pandas()
    df = df.drop_duplicates(subset="segment_path")
    keep = ["segment_path"] + cols
    renamed = {col: f"{name}_{col}" for col in cols}
    return df[keep].rename(columns=renamed)


def merge_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    return reduce(lambda left, right: left.merge(right, on="segment_path", how="inner", validate="one_to_one"), frames)


def apply_thresholds(
    merged: pd.DataFrame, thresholds: Dict[str, Dict[str, float]], teacher_names: Iterable[str], labels: Iterable[str]
) -> pd.DataFrame:
    """
    Recompute *_label columns from *_score using calibrated thresholds.
    thresholds structure: {teacher: {label: threshold}}
    """
    merged = merged.copy()
    for teacher in teacher_names:
        if teacher not in thresholds:
            continue
        for label in labels:
            key = f"{teacher}_{label}_score"
            if key not in merged.columns:
                continue
            thr = thresholds[teacher].get(label)
            if thr is None:
                continue
            merged[f"{teacher}_{label}_label"] = (merged[key] >= thr).astype(int)
    return merged


def pick_best_teachers(gold: pd.DataFrame, teacher_frames: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Choose the best teacher per label (speech/music/noise) by accuracy on the non-IRR subset only.
    """
    if "is_irr_segment" not in gold.columns:
        raise ValueError("Expected is_irr_segment column in gold annotations")
    calib = gold[~gold["is_irr_segment"]]
    if calib.empty:
        raise ValueError("No non-IRR gold rows found; cannot perform calibration")
    print(f"Using non-IRR calibration subset with {len(calib)} rows")

    merged_calib = calib.copy()
    for name, df in teacher_frames.items():
        merged_calib = merged_calib.merge(df, on="segment_path", how="left", validate="one_to_one")

    choices: Dict[str, str] = {}
    for label in ["speech", "music", "noise"]:
        best_name, best_f1 = None, -1.0
        target_col = f"{label}_gold"
        for name in teacher_frames:
            pred_col = f"{name}_{label}_label"
            tp = ((merged_calib[pred_col] == 1) & (merged_calib[target_col] == 1)).sum()
            fp = ((merged_calib[pred_col] == 1) & (merged_calib[target_col] == 0)).sum()
            fn = ((merged_calib[pred_col] == 0) & (merged_calib[target_col] == 1)).sum()
            denom = (2 * tp + fp + fn)
            f1 = 0.0 if denom == 0 else (2 * tp) / denom
            if f1 > best_f1:
                best_f1, best_name = f1, name
        assert best_name is not None
        choices[label] = best_name
        print(f"Best {label} teacher: {best_name} (F1 {best_f1:.4f})")
    return choices


def build_dataset(
    metadata_dir: Path,
    out_disk: Path | None,
    out_parquet: Path | None,
    out_csv: Path | None,
    consensus: bool = True,
    vote_threshold: int = 2,
    thresholds_path: Path | None = None,
    noise_pool: str = "mean",
) -> None:
    """
    Deterministic aggregation:
    1) Load teachers.
    2) Optional: re-threshold labels from scores using calibrated thresholds.
    3) EITHER consensus (default) via majority vote over teacher labels, or fallback to best-teacher-per-label calibration.
    4) Inner-join teachers on segment_path to ensure one row per segment.
    5) Add chosen labels/scores for speech/music/noise.
    Final dataset is teacher-only plus the chosen label/score columns.
    """
    teacher_frames = {
        name: load_teacher_df(metadata_dir, name, folder, cols) for name, (folder, cols) in TEACHER_SPECS.items()
    }

    frames: List[pd.DataFrame] = list(teacher_frames.values())
    merged = merge_frames(frames)

    # Optional: re-threshold teacher labels from scores using calibrated thresholds JSON.
    if thresholds_path:
        thresholds = json.loads(Path(thresholds_path).read_text())
        merged = apply_thresholds(merged, thresholds, teacher_frames.keys(), ["speech", "music", "noise"])
        print(f"Applied calibrated thresholds from {thresholds_path}")

    # Strict sanity checks: no duplicates and no missing segments vs the intersection of all teachers.
    for name, df in teacher_frames.items():
        if df["segment_path"].duplicated().any():
            dup_count = df["segment_path"].duplicated().sum()
            raise ValueError(f"Teacher {name} still has {dup_count} duplicate segment_path values after de-duplication step.")

    if consensus:
        labels = ["speech", "music", "noise"]
        for label in labels:
            label_cols = [f"{name}_{label}_label" for name in teacher_frames]
            score_cols = [f"{name}_{label}_score" for name in teacher_frames]
            votes = merged[label_cols].sum(axis=1)
            merged[f"chosen_{label}_label"] = (votes >= vote_threshold).astype(int)

            # Average scores over teachers that voted positive; fallback to mean of all scores if none voted.
            score_df = merged[score_cols]
            mask = merged[label_cols].astype(bool)
            pos_count = mask.sum(axis=1)
            pos_sum = (score_df * mask).sum(axis=1)
            all_mean = score_df.mean(axis=1)
            chosen_score = pos_sum.where(pos_count > 0, all_mean) / pos_count.where(pos_count > 0, 1)
            if label == "noise" and noise_pool == "max":
                chosen_score = score_df.max(axis=1)
            merged[f"chosen_{label}_score"] = chosen_score
        print(f"Applied consensus voting with threshold {vote_threshold} over teachers: {list(teacher_frames.keys())}")
    else:
        gold = pd.read_csv(metadata_dir / "blocs_smad_gold_annotations_v1.csv").drop_duplicates(subset="segment_path")
        choices = pick_best_teachers(gold, teacher_frames)
        for label in ["speech", "music", "noise"]:
            winner = choices[label]
            merged[f"chosen_{label}_label"] = merged[f"{winner}_{label}_label"]
            merged[f"chosen_{label}_score"] = merged[f"{winner}_{label}_score"]
        print(f"Used best-teacher-per-label calibration on non-IRR gold to select: {choices}")

    teacher_counts = {name: len(df) for name, df in teacher_frames.items()}
    teacher_segments = {name: set(df["segment_path"]) for name, df in teacher_frames.items()}
    intersection_size = len(set.intersection(*teacher_segments.values()))
    print(f"Teacher row counts: {teacher_counts}")
    print(f"Segment intersection size across teachers: {intersection_size}")

    if merged["segment_path"].duplicated().any():
        dup_count = merged["segment_path"].duplicated().sum()
        raise ValueError(f"Merged dataset has {dup_count} duplicate segment_path values; this should never happen.")

    if len(merged) != intersection_size:
        raise ValueError(
            f"Merged row count {len(merged)} does not match intersection size {intersection_size}. "
            "Check teacher coverage; segments may be missing in some teacher outputs."
        )

    print(f"Built merged dataset with {len(merged)} rows and {merged.shape[1]} columns")

    if out_parquet:
        merged.to_parquet(out_parquet, index=False)
        print(f"Wrote Parquet to {out_parquet}")

    if out_csv:
        merged.to_csv(out_csv, index=False)
        print(f"Wrote CSV to {out_csv}")

    if out_disk:
        ds = Dataset.from_pandas(merged, preserve_index=False)
        ds.save_to_disk(out_disk)
        print(f"Saved HF dataset to {out_disk}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SMAD finetuning dataset by merging teacher outputs (with chosen labels).")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/metadata"),
        help="Directory containing the teacher datasets and gold annotations CSV.",
    )
    parser.add_argument(
        "--consensus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use majority-vote consensus across teachers (default). If False, use best-teacher-per-label.",
    )
    parser.add_argument(
        "--vote-threshold",
        type=int,
        default=2,
        help="Votes required to set a positive label when consensus is enabled.",
    )
    parser.add_argument(
        "--thresholds-path",
        type=Path,
        default=None,
        help="Optional path to JSON with calibrated thresholds: {teacher: {label: threshold}}. If provided, *_label columns are recomputed from *_score.",
    )
    parser.add_argument(
        "--noise-pool",
        choices=["mean", "max"],
        default="mean",
        help="How to pool consensus noise scores (mean of voters or max across teachers).",
    )
    parser.add_argument(
        "--out-disk",
        type=Path,
        default=Path("data/metadata/blocs_smad_v2_finetune"),
        help="Where to save the merged dataset with Dataset.save_to_disk. Set to '' to skip.",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=Path("data/metadata/blocs_smad_v2_finetune.parquet"),
        help="Optional Parquet export of the merged table. Set to '' to skip.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/metadata/blocs_smad_v2_finetune.csv"),
        help="Optional CSV export of the merged table. Set to '' to skip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_disk = args.out_disk if str(args.out_disk) else None
    out_parquet = args.out_parquet if str(args.out_parquet) else None
    out_csv = args.out_csv if str(args.out_csv) else None
    build_dataset(
        args.metadata_dir,
        out_disk,
        out_parquet,
        out_csv,
        consensus=args.consensus,
        vote_threshold=args.vote_threshold,
        thresholds_path=args.thresholds_path,
        noise_pool=args.noise_pool,
    )


if __name__ == "__main__":
    main()
