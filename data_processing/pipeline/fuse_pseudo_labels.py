# data_processing/pipeline/fuse_pseudo_labels.py

"""Fuse multi-teacher probabilities into high-confidence pseudo-labels.

This script is intentionally policy-only: it does not run teacher inference.
It reads the merged teacher scores from v2/manifest and outputs fused labels to v3/.

Recommended use (bootstrap / first student run):
    - Speech/music: precision-targeted per-teacher thresholds + 2-of-3 agreement
      across strong foreground teachers.
    - Noise: max-pool across the two teachers that provide meaningful
      noise signal (Whisper-AT + CLAP), with a higher threshold when speech or
      music is present to keep overlap noise labels clean.
    - Co-occurrence is allowed (noise can co-occur with speech/music).

Inputs (from v2/):
    - manifest/ (HF dataset with merged teacher scores)

Expected input columns (probabilities in [0, 1]):
    ast_speech_score, ast_music_score, ast_noise_score
    whisper_speech_score, whisper_music_score, whisper_noise_score
    m2d_clap_speech_score, m2d_clap_music_score, m2d_clap_noise_score
    clap_speech_score, clap_music_score, clap_noise_score

Outputs (written to v3/):
    - manifest.csv
    - manifest/ (HF dataset)
    - fusion_policy.json
    - thresholds.csv

Output columns (hard labels 0/1 + fused scores):
    speech_pseudo, music_pseudo, noise_pseudo
    speech_score_fused, music_score_fused, noise_score_fused
    pseudo_label_source
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from datasets import Dataset, load_from_disk
import pandas as pd

from config import get_settings
from utils.artifacts import V2, V3, MANIFEST, FUSION_POLICY_JSON, FUSION_THRESHOLDS_CSV
from utils.dams_types import BatchDict
from utils.logger import logger


# ======================================================================================
# Fusion policy
# ======================================================================================

FUSION_POLICY: dict[str, Any] = {
    'version': 'v3_fused_gated_noise',
    'fg_teachers': ['ast', 'whisper', 'm2d_clap'],
    'noise_teachers': ['whisper', 'clap'],
    'thresholds': {
        'ast': {'speech': 0.60, 'music': 0.40, 'noise': 0.40},
        'whisper': {'speech': 0.60, 'music': 0.40, 'noise': 0.40},  # base value kept numeric
        'clap': {'speech': 0.45, 'music': 0.35, 'noise': 0.35},
        'm2d_clap': {'speech': 0.24, 'music': 0.26, 'noise': 0.24},
    },
    # gated whisper noise policy (numeric, exported)
    'noise_thr_no_fg': 0.40,
    'noise_thr_with_fg': 0.55,
    'label_source_value': 'fused_teachers_v3_gated_whisper_noise',
}
# Foreground teachers used for agreement voting.
FG_TEACHERS: tuple[str, ...] = tuple(FUSION_POLICY['fg_teachers'])

# Noise teachers used for max-pool union.
NOISE_TEACHERS: tuple[str, ...] = tuple(FUSION_POLICY['noise_teachers'])

# Precision-targeted thresholds for speech/music should come from the calibration
# notebook (notebooks/02_teacher_calibration.ipynb).
# The values below are sensible defaults consistent with that analysis.
THRESHOLDS: dict[str, dict[str, float]] = FUSION_POLICY['thresholds']

# Whisper noise thresholds (gated by foreground presence).
NOISE_THR_NO_FG: float = float(FUSION_POLICY['noise_thr_no_fg'])
NOISE_THR_WITH_FG: float = float(FUSION_POLICY['noise_thr_with_fg'])


def _get_score(batch: dict[str, Any], col: str, row_idx: int) -> float:
    """Read a probability score (float) from a batched HF map batch."""
    return float(batch[col][row_idx])


def _require_columns(ds: Dataset, cols: list[str]) -> None:
    """Ensure that required columns are present in the dataset."""
    missing = [c for c in cols if c not in ds.column_names]
    if missing:
        msg = (
            'Missing required columns for fusion: '
            + ', '.join(missing)
            + '.\n'
            + 'Available columns: '
            + ', '.join(ds.column_names)
        )
        raise ValueError(msg)


def _fuse_batch(batch: BatchDict) -> BatchDict:
    """Fuse teacher probabilities into hard labels."""

    # We assume segment_path exists; used only for length.
    n = len(batch['segment_path'])

    speech_out: list[int] = []
    music_out: list[int] = []
    noise_out: list[int] = []

    speech_score_fused: list[float] = []
    music_score_fused: list[float] = []
    noise_score_fused: list[float] = []

    pseudo_label_source: list[str] = []

    for row_idx in range(n):
        # ==============================================================================
        # Speech: 2-of-3 agreement over FG teachers
        # ==============================================================================
        speech_votes = 0
        s_scores: list[float] = []
        for teacher in FG_TEACHERS:
            col = f'{teacher}_speech_score'
            s = _get_score(batch, col, row_idx)
            s_scores.append(s)
            speech_votes += int(s >= THRESHOLDS[teacher]['speech'])

        speech_label = int(speech_votes >= 2)
        speech_out.append(speech_label)
        speech_score_fused.append(max(s_scores))

        # ==============================================================================
        # Music: 2-of-3 agreement over FG teachers
        # ==============================================================================
        music_votes = 0
        m_scores: list[float] = []
        for teacher in FG_TEACHERS:
            col = f'{teacher}_music_score'
            m = _get_score(batch, col, row_idx)
            m_scores.append(m)
            music_votes += int(m >= THRESHOLDS[teacher]['music'])

        music_label = int(music_votes >= 2)
        music_out.append(music_label)
        music_score_fused.append(max(m_scores))

        # ==============================================================================
        # Noise: OR of teacher votes, with gated Whisper threshold when foreground is present
        # ==============================================================================
        whisper_noise = _get_score(batch, 'whisper_noise_score', row_idx)
        clap_noise = _get_score(batch, 'clap_noise_score', row_idx)

        # Keep a fused score for analysis/debugging.
        noise_score_fused.append(max(whisper_noise, clap_noise))

        # Foreground is defined by the fused speech/music labels.
        fg_on = (speech_label == 1) or (music_label == 1)

        # Whisper noise uses a gated threshold: stricter when foreground is present.
        thr_whisper_noise = NOISE_THR_WITH_FG if fg_on else NOISE_THR_NO_FG
        whisper_noise_vote = int(whisper_noise >= thr_whisper_noise)

        # CLAP uses a single fixed noise threshold.
        clap_noise_vote = int(clap_noise >= THRESHOLDS['clap']['noise'])

        # Final noise label is the union (OR) of teacher votes.
        noise_label = int((whisper_noise_vote == 1) or (clap_noise_vote == 1))
        noise_out.append(noise_label)

        pseudo_label_source.append(str(FUSION_POLICY['label_source_value']))

    batch['speech_pseudo'] = speech_out
    batch['music_pseudo'] = music_out
    batch['noise_pseudo'] = noise_out

    batch['speech_score_fused'] = speech_score_fused
    batch['music_score_fused'] = music_score_fused
    batch['noise_score_fused'] = noise_score_fused

    batch['pseudo_label_source'] = pseudo_label_source

    return batch

def export_fusion_policy(v3_path: Path) -> None:
    """Export the fusion policy as JSON and CSV for reproducibility and logging."""

    # Prepare output paths for JSON.
    out_json = v3_path / FUSION_POLICY_JSON
    out_json.write_text(json.dumps(FUSION_POLICY, indent=2), encoding='utf-8')

    # Prepare rows for CSV export.
    rows: list[dict[str, Any]] = []
    for teacher, tmap in THRESHOLDS.items():
        for label, thr in tmap.items():
            row: dict[str, Any] = {
                'teacher': teacher,
                'label': label,
                'new_threshold': float(thr),
                'noise_thr_no_fg': None,
                'noise_thr_with_fg': None,
            }
            if label == 'noise' and teacher == 'whisper':
                row['noise_thr_no_fg'] = float(NOISE_THR_NO_FG)
                row['noise_thr_with_fg'] = float(NOISE_THR_WITH_FG)
            rows.append(row)

    # Ensure whisper noise gating thresholds are always explicit in the CSV.
    # This adds a dedicated row even if the thresholds map changes later.
    rows.append(
        {
            'teacher': 'whisper',
            'label': 'noise',
            'new_threshold': float(THRESHOLDS['whisper']['noise']),
            'noise_thr_no_fg': float(NOISE_THR_NO_FG),
            'noise_thr_with_fg': float(NOISE_THR_WITH_FG),
        }
    )

    out_csv = v3_path / FUSION_THRESHOLDS_CSV
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    logger.info(f'Exported fusion policy JSON: {out_json}')
    logger.info(f'Exported fusion policy CSV: {out_csv}')



def main() -> None:

    settings = get_settings()
    v2_path: Path = settings.manifest_dir(V2)
    v3_path: Path = settings.manifest_dir(V3)
    v3_path.mkdir(parents=True, exist_ok=True)

    export_fusion_policy(v3_path)

    # Load merged teacher scores from v2/manifest.
    in_path = v2_path / MANIFEST
    logger.info(f'Loading teacher scores from {in_path}...')
    ds: Dataset = load_from_disk(in_path)

    required_cols = [
        'segment_path',
        'ast_speech_score',
        'ast_music_score',
        'whisper_speech_score',
        'whisper_music_score',
        'whisper_noise_score',
        'm2d_clap_speech_score',
        'm2d_clap_music_score',
        'clap_noise_score',
    ]
    _require_columns(ds, required_cols)

    optional_cols = ['raw_file', 'start_time', 'end_time', 'split']
    missing_optional = [c for c in optional_cols if c not in ds.column_names]
    if missing_optional:
        logger.warning(
             f'Missing optional columns: {", ".join(missing_optional)}. '
            'Fusion will proceed, but assign_splits needs timing metadata.'
        )

    logger.info(
        'Fusing labels: speech/music via 2-of-3 agreement (AST+Whisper+M2D), '
        'noise via max(Whisper, CLAP) with overlap-aware threshold...'
    )
    ds_fused = ds.map(
        _fuse_batch,
        batched=True,
        batch_size=1024,
        desc='Fusing teacher labels',
    )

    # Save to v3/manifest.csv and v3/manifest/
    ds_fused.to_csv(v3_path / f'{MANIFEST}.csv', index=False)
    ds_fused.save_to_disk(v3_path / MANIFEST)
    logger.info(f'✓ Saved fused manifest to {v3_path / MANIFEST}[.csv]')


if __name__ == '__main__':
    main()
