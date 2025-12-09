# data_processing/teachers/apply_whisper.py

"""
Apply Whisper-AT teacher model to generate pseudo-labels for audio segments.

Reads audio segments from disk, runs Whisper-AT to obtain AudioSet logits,
collapses them to speech, music, and noise scores, and saves pseudo-labels
back to disk.

Usage:
    python -m data_processing.teachers.apply_whisper
"""

from pathlib import Path

from functools import partial

import torch
from datasets import Dataset

import whisper_at as whisper

from config import get_settings, SAMPLE_RATE, WHISPER_MODEL_SIZE

from utils.dams_types import (
    BatchDict,
    SEGMENT_PATH,
    LABEL_SOURCE_WHISPER_PSEUDO,
    SPEECH,
    MUSIC,
    NOISE,
    SPEECH_SCORE,
    MUSIC_SCORE,
    NOISE_SCORE,
    LABEL_SOURCE_FIELD,
    BLOCS_SMAD_V1, BLOCS_SMAD_V2_WHISPER,
)

from utils.audioset_mapping import (
    SPEECH_IDX,
    MUSIC_IDX,
    NOISE_IDX,
    AST_POOLING,
    AST_SPEECH_THRESHOLD,
    AST_MUSIC_THRESHOLD,
    collapse_audioset_logits,
)

from utils.logger import logger, log_pseudo_label_stats


# Whisper-AT config for this project.
# Segments are 10 s with 50% overlap, so we match that for tagging.
WHISPER_MODEL_SIZE = 'large-v2'
WHISPER_AT_TIME_RES = 10.0  # seconds, hop and window for audio_tagging all the same.


def _load_whisper_model() -> "whisper.WhisperAT":
    """
    Load Whisper-AT once at module import time.

    We keep it simple here and let whisper_at choose the device,
    since internally it mirrors the original Whisper API.
    """
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    logger.info(f'Loading Whisper-AT model: {WHISPER_MODEL_SIZE} on {device}...')
    model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
    model.eval()
    return model


whisper_model = _load_whisper_model()


def _apply_whisper_to_batch(batch: BatchDict, segments_dir: Path) -> BatchDict:
    """
    Apply Whisper-AT to a batch of segment files.

    For each segment we:
      - call Whisper-AT with at_time_res equal to the segment length
      - take the first AudioSet logit vector (since the clip is one segment)
      - apply sigmoid to get per-class probabilities
      - collapse to pooled speech/music/noise scores
      - threshold to obtain binary labels
    """
    segment_names: list[str] = batch[SEGMENT_PATH]

    whisper_probs_batch: list[list[float]] = []
    speech_labels: list[int] = []
    music_labels: list[int] = []
    noise_labels: list[int] = []
    speech_scores: list[float] = []
    music_scores: list[float] = []
    noise_scores: list[float] = []

    for name in segment_names:
        seg_path = segments_dir / name

        # Whisper-AT API; it handles decoding and resampling internally.
        # at_time_res controls the hop and window for audio_tag.
        result = whisper_model.transcribe(
            str(seg_path),
            at_time_res=WHISPER_AT_TIME_RES,
            fp16=False,  # avoid FP16 on CPU / MPS and kill that warning.
        )

        audio_tag = result['audio_tag']  # shape [T, 527], unnormalised logits

        if isinstance(audio_tag, torch.Tensor):
            logits_vec: torch.Tensor
            if audio_tag.ndim == 2:
                # For a 10 s clip and at_time_res=10, this should be [1, 527]
                logits_vec = audio_tag[0]
            else:
                # Fallback; treat as a single 527 vector
                logits_vec = audio_tag.view(-1)
        else:
            # In case library returns a list-like, convert to tensor.
            logits_vec = torch.as_tensor(audio_tag, dtype=torch.float32)
            if logits_vec.ndim == 2:
                logits_vec = logits_vec[0]

        # Store per-class probabilities for downstream analysis.
        probs_vec = logits_vec.sigmoid()
        whisper_probs_batch.append(probs_vec.tolist())

        # Collapse AudioSet logits into pooled speech, music, noise scores.
        # collapse_audioset_logits expects shape [batch, 527].
        pooled_scores = collapse_audioset_logits(
            logits_vec.unsqueeze(0),
            speech_idx=SPEECH_IDX,
            music_idx=MUSIC_IDX,
            noise_idx=NOISE_IDX,
            pooling=AST_POOLING,  # reuse pooling strategy for now
        ).squeeze(0)

        s_score = float(pooled_scores[0].item())
        m_score = float(pooled_scores[1].item())
        n_score = float(pooled_scores[2].item())

        speech_scores.append(s_score)
        music_scores.append(m_score)
        noise_scores.append(n_score)

        # Thresholds: reuse AST thresholds as a starting point.
        s_label = int(s_score >= AST_SPEECH_THRESHOLD)
        m_label = int(m_score >= AST_MUSIC_THRESHOLD)

        # Residual noise, only when neither speech nor music is active.
        n_label = int((s_label == 0) and (m_label == 0))

        speech_labels.append(s_label)
        music_labels.append(m_label)
        noise_labels.append(n_label)

    batch[SPEECH] = speech_labels
    batch[MUSIC] = music_labels
    batch[NOISE] = noise_labels
    batch[SPEECH_SCORE] = speech_scores
    batch[MUSIC_SCORE] = music_scores
    batch[NOISE_SCORE] = noise_scores

    # Store the full 527-way probabilities from Whisper-AT.
    batch['whisper_probs'] = whisper_probs_batch

    # Mark label source.
    batch[LABEL_SOURCE_FIELD] = [LABEL_SOURCE_WHISPER_PSEUDO] * len(segment_names)

    return batch


def main() -> None:
    settings = get_settings()
    metadata_dir: Path = settings.metadata_path
    segments_dir: Path = settings.segments_path

    # You can change this to BLOCS_SMAD_V2 or BLOCS_SMAD_V2_GOLD
    # if you want to start from the AST-labeled or gold-aligned dataset.
    base_manifest_path = metadata_dir / BLOCS_SMAD_V1
    logger.info(f'Loading base dataset from {base_manifest_path}...')
    ds: Dataset = Dataset.load_from_disk(base_manifest_path)

    map_fn = partial(_apply_whisper_to_batch, segments_dir=segments_dir)

    logger.info('Applying Whisper-AT teacher...')
    ds_whisper = ds.map(
        map_fn,
        batched=True,
        batch_size=1,   # Whisper-AT API is per-clip; keep batch size 1
        desc='Whisper-AT pseudo labeling',
    )

    out_name = BLOCS_SMAD_V2_WHISPER
    out_path = metadata_dir / out_name

    logger.info(f'Saving Whisper-AT labeled dataset to {out_path}...')
    ds_whisper.save_to_disk(out_path)
    ds_whisper.to_csv(str(metadata_dir / f'{out_name}.csv'), index=False)

    log_pseudo_label_stats(ds_whisper, teacher_name="Whisper-AT pseudo teacher")


if __name__ == '__main__':
    main()
