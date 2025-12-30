# data_processing/merge_teacher_scores.py

"""Build a unified, per-segment teacher score table.

Merges per-teacher outputs from v2/teachers/ into a single manifest at v2/manifest.csv.
fuse_pseudo_labels.py expects this unified table as input.

Inputs (HF datasets from v2/teachers/):
  - ast/
  - clap/
  - m2d/
  - whisper/

Outputs (written to v2/):
  - manifest.csv
  - manifest/ (HF dataset)

Expected output columns:
  segment_path, raw_file, start_time, end_time,
  ast_speech_score, ast_music_score, ast_noise_score,
  clap_speech_score, clap_music_score, clap_noise_score,
  m2d_clap_speech_score, m2d_clap_music_score, m2d_clap_noise_score,
  whisper_speech_score, whisper_music_score, whisper_noise_score

Notes:
  - This script performs an inner join on segment_path across teachers.
  - If any teacher is missing segments, the output row count will shrink.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset, load_from_disk

from config import get_settings

from utils.artifacts import V1, V2, SEGMENTS_CSV, TEACHERS_DIR, MANIFEST

from utils.logger import logger

# Maps output prefix -> directory name in v2/teachers/.
TEACHER_MAP = {
    'ast': 'ast',
    'clap': 'clap',
    'm2d_clap': 'm2d',
    'whisper': 'whisper',
}


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def make_prefixed_teacher_df(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Keep segment_path and SMN scores, prefix scores with teacher shortname."""
    cols = ['segment_path', 'speech_score', 'music_score', 'noise_score']
    _require_columns(df, cols, prefix)

    df_sub = df[cols].copy()
    rename_map = {
        'speech_score': f'{prefix}_speech_score',
        'music_score': f'{prefix}_music_score',
        'noise_score': f'{prefix}_noise_score',
    }
    return df_sub.rename(columns=rename_map)


def build_teacher_scores(v1_dir: Path, teachers_path: Path) -> Dataset:
    """Load per teacher datasets, merge on segment_path, and return a unified Dataset."""

    # Load base segments for metadata columns.
    segments_df = pd.read_csv(v1_dir / SEGMENTS_CSV)
    logger.info(f"Loaded segments: {len(segments_df)} rows")

    # Load each teacher's HF dataset.
    teacher_dfs = {}
    for prefix, dirname in TEACHER_MAP.items():
        ds_path = teachers_path / dirname
        if not ds_path.exists():
            raise FileNotFoundError(f'Missing teacher output: {ds_path}')
        teacher_dfs[prefix] = load_from_disk(ds_path).to_pandas()
        logger.info(f"  {prefix} (from {dirname}/): {len(teacher_dfs[prefix])} rows")

    # Prefix score columns
    prefixed = {
        prefix: make_prefixed_teacher_df(df, prefix)
        for prefix, df in teacher_dfs.items()
    }

    # Start with segments, merge each teacher
    merged = segments_df[['segment_path', 'raw_file', 'start_time', 'end_time']].copy()
    for prefix, df in prefixed.items():
        merged = merged.merge(df, on='segment_path', how='inner')

    # Check for duplicates
    dup = int(merged['segment_path'].duplicated().sum())
    if dup:
        raise ValueError(f'Merged table has {dup} duplicate segment_path rows')

    # Optional: sanity check for fusion script
    required = [
        'segment_path', 'raw_file', 'start_time', 'end_time',
        'ast_speech_score', 'ast_music_score', 'ast_noise_score',
        'clap_speech_score', 'clap_music_score', 'clap_noise_score',
        'm2d_clap_speech_score', 'm2d_clap_music_score', 'm2d_clap_noise_score',
        'whisper_speech_score', 'whisper_music_score', 'whisper_noise_score',
    ]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise ValueError(
            f'Missing columns required by fuse_pseudo_labels.py: {missing}')

    logger.info(f'Merged teacher score table: {merged.shape}')

    return Dataset.from_pandas(merged, preserve_index=False)


def main() -> None:

    settings = get_settings()
    v1_path = settings.manifest_dir(V1)
    v2_path = settings.manifest_dir(V2)
    teachers_path = v2_path / TEACHERS_DIR

    logger.info(f"Loading teachers from: {teachers_path}")

    ds = build_teacher_scores(v1_path, teachers_path)

    # Save to v2/manifest.csv and v2/manifest/
    ds.to_csv(v2_path / f'{MANIFEST}.csv', index=False)
    ds.save_to_disk(v2_path / MANIFEST)
    logger.info(f'✓ Saved merged manifest to {v2_path / MANIFEST}[.csv]')

    logger.info('Done.')


if __name__ == '__main__':
    main()
