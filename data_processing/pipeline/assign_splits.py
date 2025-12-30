# data_processing/pipeline/assign_splits.py

"""Assign leakage-safe splits for BLOCS-SMAD and build a training manifest.

This script:
  1) Joins gold annotations onto the base segment manifest.
  2) Assigns calibration/train/dev/test splits with leakage controls.
  3) Optionally joins fused pseudo-labels (from `fuse_pseudo_labels.py`).
  4) Emits a final manifest where gold overrides pseudo.

Split policy:
  - calibration: IRR gold only
  - dev/test: gold only
  - train: gold + pseudo-only segments in train blocks
  - unlabeled: pseudo-only segments not selected for train
"""

from __future__ import annotations

import sys
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit

from config import get_settings

from utils.artifacts import (
    V1, V3, V4,
    SEGMENTS_CSV,
    MANIFEST,
    GOLD_ANNOTATIONS_CSV,
    SPLIT_MAP_CSV, SPLIT_LOG_JSON
)

from utils.dams_types import (
    REQUIRED_BASE_COLS,
    REQUIRED_GOLD_COLS,
    REQUIRED_PSEUDO_COLS,
)

from utils.logger import logger

@dataclass(frozen=True)
class SplitFractions:
    train: float
    dev: float
    test: float

    def validate(self) -> None:
        total = self.train + self.dev + self.test
        if not np.isclose(total, 1.0):
            raise ValueError(f'Split fractions must sum to 1.0, got {total}')
        for name, v in [('train', self.train), ('dev', self.dev), ('test', self.test)]:
            if v <= 0.0 or v >= 1.0:
                raise ValueError(f'Invalid fraction for {name}: {v}')


def _ensure_cols(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')


def _safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().map({'true': True, 'false': False}).fillna(False)


def _summarize_gold(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out['n_total'] = int(len(df))
    out['n_irr'] = int(df['is_irr_segment'].sum())
    out['n_non_irr'] = int((~df['is_irr_segment']).sum())

    for lab in ['speech', 'music', 'noise']:
        col = f'{lab}_gold'
        out[f'{lab}_pos'] = int(df[col].sum())
        out[f'{lab}_pos_rate'] = float(df[col].mean()) if len(df) else 0.0

    return out


def _write_split_log(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def _make_group_id(
    df: pd.DataFrame,
    group_mode: Literal['raw_file', 'time_block'],
    block_seconds: float,
) -> pd.Series:
    """Return a grouping key used to prevent leakage across splits.

    `raw_file` keeps whole recordings together.
    `time_block` keeps segments from the same recording and nearby time together,
    which is useful when you have only a few long recordings.
    """
    if group_mode == 'raw_file':
        return df['raw_file'].astype(str)

    # time_block
    if block_seconds <= 0:
        raise ValueError(f'block_seconds must be > 0, got {block_seconds}')
    if 'start_time' not in df.columns:
        raise ValueError('start_time column is required for time_block grouping')

    block_idx = (df['start_time'].astype(float) // float(block_seconds)).astype(int)
    return df['raw_file'].astype(str) + '::' + block_idx.astype(str)


def _group_split_gold_non_irr(
    gold_non_irr: pd.DataFrame,
    fracs: SplitFractions,
    seed: int,
    group_mode: Literal['raw_file', 'time_block'],
    block_seconds: float,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    fracs.validate()

    groups = _make_group_id(gold_non_irr, group_mode=group_mode, block_seconds=block_seconds)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=fracs.test, random_state=seed)
    train_dev_idx, test_idx = next(gss1.split(gold_non_irr, groups=groups))

    train_dev = gold_non_irr.iloc[train_dev_idx]

    # dev should be fracs.dev of the full set. Convert to a fraction of the remaining.
    dev_frac_of_train_dev = fracs.dev / (fracs.train + fracs.dev)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=dev_frac_of_train_dev, random_state=seed)
    train_idx, dev_idx = next(
        gss2.split(
            train_dev,
            groups=_make_group_id(train_dev, group_mode=group_mode, block_seconds=block_seconds),
        )
    )

    train_index = train_dev.iloc[train_idx].index
    dev_index = train_dev.iloc[dev_idx].index
    test_index = gold_non_irr.iloc[test_idx].index

    return train_index, dev_index, test_index


def _build_block_columns(df: pd.DataFrame, block_seconds: float) -> pd.DataFrame:
    """Add block_idx and block_id columns for time-block grouping."""
    out = df.copy()
    out['block_idx'] = (out['start_time'].astype(float) // float(block_seconds)).astype(int)
    out['block_id'] = out['raw_file'].astype(str) + '::' + out['block_idx'].astype(str)
    return out


def _apply_neighbor_buffer(
    blocks: pd.DataFrame,
    all_blocks_by_raw: dict[str, set[int]],
    buffer_blocks: int,
) -> dict[tuple[str, int], str]:
    """Expand train/dev/test assignments to neighboring time blocks within each raw_file.

    This prevents leakage when segments overlap across block boundaries.

    Important: this expansion is **non-transitive**.
    We expand only around the originally assigned blocks, not around newly expanded blocks.

    Priority order resolves conflicts deterministically: test > dev > train.
    Calibration is handled separately at the segment level and is not modified here.
    """
    if len(blocks) == 0:
        return {}

    priority = {'train': 1, 'dev': 2, 'test': 3}

    # Seed map from the initial assignment (gold non-IRR only)
    seed_map: dict[tuple[str, int], str] = {}
    for r, b, s in blocks[['raw_file', 'block_idx', 'split']].itertuples(index=False, name=None):
        key = (str(r), int(b))
        split = str(s)
        cur = seed_map.get(key)
        if cur is None or priority.get(split, 0) > priority.get(cur, 0):
            seed_map[key] = split

    if buffer_blocks <= 0:
        return dict(seed_map)

    expanded: dict[tuple[str, int], str] = dict(seed_map)

    # Expand only around the original seed blocks (one hop), not iteratively.
    for (raw, blk), split in seed_map.items():
        if split not in priority:
            continue

        for nb in range(int(blk) - int(buffer_blocks), int(blk) + int(buffer_blocks) + 1):
            if nb < 0:
                continue
            if nb not in all_blocks_by_raw.get(raw, set()):
                continue

            key = (raw, int(nb))
            cur = expanded.get(key)
            if cur is None or priority.get(split, 0) > priority.get(cur, 0):
                expanded[key] = split

    return expanded


def assign_splits(
    base_manifest_csv: Path,
    gold_csv: Path,
    out_dir: Path,
    fracs: SplitFractions,
    seed: int,
    group_mode: Literal['raw_file', 'time_block'],
    block_seconds: float,
    buffer_blocks: int,
    fused_pseudo_csv: Path | None,
) -> None:
    base = pd.read_csv(base_manifest_csv)
    gold = pd.read_csv(gold_csv)

    _ensure_cols(base, REQUIRED_BASE_COLS, name='base manifest')
    _ensure_cols(gold, REQUIRED_GOLD_COLS, name='gold annotations')

    pseudo: pd.DataFrame | None = None
    if fused_pseudo_csv is not None:
        pseudo = pd.read_csv(fused_pseudo_csv)
        _ensure_cols(pseudo, REQUIRED_PSEUDO_COLS, name='fused pseudo labels')

    if group_mode == 'time_block' and block_seconds <= 0:
        raise ValueError(f'block_seconds must be > 0 for time_block, got {block_seconds}')

    gold = gold.copy()
    gold['is_irr_segment'] = _safe_bool_series(gold['is_irr_segment'])

    for lab in ['speech', 'music', 'noise']:
        gold[f'{lab}_gold'] = gold[f'{lab}_gold'].astype(int)

    # Join gold onto base by segment_path.
    gold_sub = gold[[
        'segment_path',
        'raw_file',
        'is_irr_segment',
        'speech_gold',
        'music_gold',
        'noise_gold',
    ]].copy()

    merged = base.merge(gold_sub, on='segment_path', how='left', suffixes=('', '_goldfile'))

    if pseudo is not None:
        pseudo_sub = pseudo[REQUIRED_PSEUDO_COLS].copy()
        merged = merged.merge(pseudo_sub, on='segment_path', how='left')

    # Determine gold membership.
    merged['is_gold'] = merged['speech_gold'].notna()
    merged['is_irr_segment'] = _safe_bool_series(merged['is_irr_segment']).astype(bool)

    # Determine whether this row has pseudo labels available.
    if 'speech_pseudo' in merged.columns:
        merged['has_pseudo'] = merged['speech_pseudo'].notna()
    else:
        merged['has_pseudo'] = False

    if group_mode == 'time_block':
        merged = _build_block_columns(merged, block_seconds=block_seconds)

    # Default split assignment.
    merged['split'] = 'unlabeled'

    # Calibration split is IRR gold.
    merged.loc[merged['is_gold'] & merged['is_irr_segment'], 'split'] = 'calibration'

    # Student gold splits come from non IRR gold only.
    gold_non_irr = merged[merged['is_gold'] & (~merged['is_irr_segment'])].copy()

    train_idx, dev_idx, test_idx = _group_split_gold_non_irr(
        gold_non_irr,
        fracs=fracs,
        seed=seed,
        group_mode=group_mode,
        block_seconds=block_seconds,
    )

    # Initial split assignment on the gold non-IRR subset.
    merged.loc[train_idx, 'split'] = 'train'
    merged.loc[dev_idx, 'split'] = 'dev'
    merged.loc[test_idx, 'split'] = 'test'

    # If using time blocks, expand train/dev/test to neighboring blocks to prevent
    # overlap leakage across boundaries (10s windows with 5s hop).
    if group_mode == 'time_block':
        # Build the set of all blocks present per raw file (across all segments).
        all_blocks_by_raw: dict[str, set[int]] = defaultdict(set)
        for r, b in merged[['raw_file', 'block_idx']].dropna().itertuples(index=False, name=None):
            all_blocks_by_raw[str(r)].add(int(b))

        # Build a unique block-level assignment from the current row-level labels.
        block_assign = (
            merged[merged['split'].isin(['train', 'dev', 'test'])]
            [['raw_file', 'block_idx', 'split']]
            .drop_duplicates(subset=['raw_file', 'block_idx'])
            .copy()
        )

        expanded = _apply_neighbor_buffer(
            block_assign,
            all_blocks_by_raw=all_blocks_by_raw,
            buffer_blocks=int(buffer_blocks),
        )

        # Apply expanded block assignments with the policy:
        # - calibration: IRR gold only
        # - dev/test: gold segments only
        # - train: gold + non-gold (pseudo-only) segments that fall in train blocks
        def _lookup_block_split(row: pd.Series) -> str | None:
            key = (str(row['raw_file']), int(row['block_idx']))
            return expanded.get(key)

        block_split_series = merged.apply(_lookup_block_split, axis=1)

        # Never overwrite calibration rows.
        mask_not_cal = merged['split'] != 'calibration'

        # Train can include non-gold (pseudo-only) segments.
        train_mask = mask_not_cal & (block_split_series == 'train')
        merged.loc[train_mask, 'split'] = 'train'

        # Dev/test are restricted to gold only.
        dev_mask = mask_not_cal & merged['is_gold'] & (block_split_series == 'dev')
        test_mask = mask_not_cal & merged['is_gold'] & (block_split_series == 'test')
        merged.loc[dev_mask, 'split'] = 'dev'
        merged.loc[test_mask, 'split'] = 'test'

        # Non-gold segments that fall into dev/test blocks remain unlabeled.
        non_gold_eval_mask = mask_not_cal & (~merged['is_gold']) & block_split_series.isin(['dev', 'test'])
        merged.loc[non_gold_eval_mask, 'split'] = 'unlabeled'

        # Use *all* pseudo-only segments for training, except those in eval/calibration blocks.
        # This keeps evaluation leakage-safe while maximizing training data.
        if 'has_pseudo' in merged.columns:
            pseudo_train_mask = (
                mask_not_cal
                & (~merged['is_gold'])
                & merged['has_pseudo']
                & (~block_split_series.isin(['dev', 'test']))
            )
            merged.loc[pseudo_train_mask, 'split'] = 'train'

    # Build final training labels.
    # Gold overrides pseudo. Rows without gold or pseudo keep 0 labels and remain unlabeled.
    for lab in ['speech', 'music', 'noise']:
        gold_col = f'{lab}_gold'
        pseudo_col = f'{lab}_pseudo'
        out_col = f'{lab}_label'

        if pseudo_col in merged.columns:
            merged[out_col] = np.where(
                merged['is_gold'],
                merged[gold_col].fillna(0).astype(int),
                merged[pseudo_col].fillna(0).astype(int),
            )
        else:
            merged[out_col] = np.where(
                merged['is_gold'],
                merged[gold_col].fillna(0).astype(int),
                0,
            ).astype(int)

    # Attach a single label source for the final labels.
    if 'pseudo_label_source' in merged.columns:
        merged['label_source'] = np.where(
            merged['is_gold'],
            'gold',
            merged['pseudo_label_source'].fillna('none').astype(str),
        )
    else:
        merged['label_source'] = np.where(merged['is_gold'], 'gold', 'none')

    # Provide fused scores for analysis, fall back to NaN when absent.
    if 'speech_score_fused' in merged.columns:
        merged['speech_score'] = merged['speech_score_fused']
        merged['music_score'] = merged['music_score_fused']
        merged['noise_score'] = merged['noise_score_fused']
    else:
        merged['speech_score'] = np.nan
        merged['music_score'] = np.nan
        merged['noise_score'] = np.nan

    # Save outputs to v4/
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV output.
    merged.to_csv(out_dir / f'{MANIFEST}.csv', index=False)

    # HF dataset output.
    hf_ds = Dataset.from_pandas(merged, preserve_index=False)
    hf_ds.save_to_disk(out_dir / MANIFEST)

    # Split map
    merged[['segment_path', 'raw_file', 'split', 'is_gold', 'is_irr_segment']].to_csv(
        out_dir / SPLIT_MAP_CSV, index=False
    )

    # Print summary
    gold_only = merged[merged['is_gold']].copy()
    print('=== GOLD label summary (joined) ===')
    print(_summarize_gold(gold_only))

    for name in ['calibration', 'train', 'dev', 'test']:
        part = merged[(merged['split'] == name) & (merged['is_gold'])].copy()
        if len(part) == 0:
            print(f'[{name}] n=0')
            continue
        stats = _summarize_gold(part)
        print(f'[{name}] {stats}')

    unl = merged[merged['split'] == 'unlabeled']
    print(f'[unlabeled] n={int(len(unl))}')

    rows_per_split = merged['split'].value_counts(dropna=False).to_dict()
    rows_per_split_is_gold = (
        merged.groupby(['split', 'is_gold']).size().to_dict()
    )

    payload: dict[str, Any] = {
        'base_manifest_csv': str(base_manifest_csv),
        'gold_csv': str(gold_csv),
        'fused_pseudo_csv': str(fused_pseudo_csv) if fused_pseudo_csv is not None else None,
        'out_manifest_csv': str(out_dir),
        'seed': int(seed),
        'group_mode': str(group_mode),
        'block_seconds': float(block_seconds),
        'buffer_blocks': int(buffer_blocks),
        'split_fracs': {'train': float(fracs.train), 'dev': float(fracs.dev), 'test': float(fracs.test)},
        'rows_per_split': {str(k): int(v) for k, v in rows_per_split.items()},
        'rows_per_split_is_gold': {str(k): int(v) for k, v in rows_per_split_is_gold.items()},
        'gold_summary_joined': _summarize_gold(merged[merged['is_gold']].copy()),
    }
    _write_split_log(out_dir / SPLIT_LOG_JSON, payload)
    logger.info(f'✓ Saved split log to {out_dir / SPLIT_LOG_JSON}')


def main() -> None:

    if len(sys.argv) == 1:
        settings = get_settings()
        v1_path = settings.manifest_dir(V1)
        v3_path = settings.manifest_dir(V3)
        v4_path = settings.manifest_dir(V4)

        logger.info('Running with default versioned paths...')

        assign_splits(
            base_manifest_csv=v1_path / SEGMENTS_CSV,
            gold_csv=v1_path / GOLD_ANNOTATIONS_CSV,
            out_dir=v4_path,
            fracs=SplitFractions(train=0.60, dev=0.20, test=0.20),
            seed=1337,
            group_mode='time_block',
            block_seconds=600.0,
            buffer_blocks=1,
            fused_pseudo_csv=v3_path / f'{MANIFEST}.csv',
        )
        return

    # Otherwise, parse CLI args.
    p = argparse.ArgumentParser(
        description='Assign train/dev/test splits to BLOCS SMAD using grouped splits'
                    ' to prevent leakage.'
    )
    p.add_argument('--base_manifest_csv', type=str, required=True)
    p.add_argument('--gold_csv', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--fused_pseudo_csv', type=str, default=None)

    p.add_argument('--train_frac', type=float, default=0.60)
    p.add_argument('--dev_frac', type=float, default=0.20)
    p.add_argument('--test_frac', type=float, default=0.20)
    p.add_argument('--seed', type=int, default=1337)

    p.add_argument(
        '--group_mode', type=str, default='time_block',
        choices=['raw_file', 'time_block']
    )
    p.add_argument('--block_seconds', type=float, default=600.0)
    p.add_argument('--buffer_blocks', type=int, default=1)

    args = p.parse_args()

    fracs = SplitFractions(
        train=float(args.train_frac),
        dev=float(args.dev_frac),
        test=float(args.test_frac)
    )

    assign_splits(
        base_manifest_csv=Path(args.base_manifest_csv),
        gold_csv=Path(args.gold_csv),
        out_dir=Path(args.out_dir),
        fracs=fracs,
        seed=int(args.seed),
        group_mode=str(args.group_mode),
        block_seconds=float(args.block_seconds),
        buffer_blocks=int(args.buffer_blocks),
        fused_pseudo_csv=Path(args.fused_pseudo_csv) if args.fused_pseudo_csv else None,
    )

if __name__ == '__main__':
    main()
