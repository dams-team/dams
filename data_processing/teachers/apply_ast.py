# data_processing/apply_ast.py

"""
Apply Supervised AudioSet Teacher model to generate pseudo-labels for audio segments.

Reads audio segments from disk, processes them with a pretrained AudioSet
Transformer (AST) model to obtain pseudo-labels for speech, music, and noise,
and saves the labeled dataset back to disk.

Usage:
    python -m data_processing.teachers.apply_ast
"""

from pathlib import Path

from functools import partial

import torch
import torchaudio
import pandas as pd
from datasets import Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from config import get_settings, AST_MODEL_NAME, SAMPLE_RATE

from utils.dams_types import (
    BatchDict,
    SEGMENT_PATH,
    LABEL_SOURCE_AST_PSEUDO,
    SPEECH,
    MUSIC,
    NOISE,
    SPEECH_SCORE,
    MUSIC_SCORE,
    NOISE_SCORE,
    LABEL_SOURCE_FIELD,
    BLOCS_SMAD_V1,
    BLOCS_SMAD_V2_AST,
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
from utils.timing import time_block

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

device = get_device()
feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
ast_model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_NAME).to(device)
ast_model.eval()  # set model to evaluation mode.


def _load_segment(segments_dir: Path, segment_name: str) -> torch.Tensor:
    seg_path = segments_dir / segment_name
    waveform, sr = torchaudio.load(seg_path)

    assert sr == SAMPLE_RATE, f"Unexpected sample rate for {segment_name}: {sr}"
    assert waveform.size(0) == 1, f"Expected mono: found {waveform.size(0)} channels"

    return waveform.squeeze(0)  # Return mono 1D tensor, shape [samples]


def _apply_ast_to_batch(batch: BatchDict, segments_dir: Path) -> BatchDict:
    segment_names: list[str] = batch[SEGMENT_PATH]

    waveforms = [ _load_segment(segments_dir, name) for name in segment_names ]
    sampling_rate = SAMPLE_RATE

    # Transformer feature extractor expects numpy arrays.
    inputs = feature_extractor(
        [wf.numpy() for wf in waveforms],
        sampling_rate=sampling_rate,
        return_tensors='pt',
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = ast_model(**inputs)
        logits_batch: torch.Tensor = outputs.logits  # shape [batch, 527]

    # Convert raw logits to per-class probabilities and store them
    # for downstream analysis. Each row has AST_NUM_LABELS entries
    # corresponding to the AudioSet ontology.
    probs_batch = logits_batch.sigmoid().cpu()  # shape [batch, 527], values in [0, 1]
    batch['ast_probs'] = probs_batch.tolist()  # list[list[float]] of length 527.

    pooled_scores = collapse_audioset_logits(
        logits_batch,
        speech_idx=SPEECH_IDX,
        music_idx=MUSIC_IDX,
        noise_idx=NOISE_IDX,
        pooling=AST_POOLING
    ).cpu()  # shape [batch, 3], values in [0, 1]

    # Extract individual class scores from tensors and combine into binary labels.
    speech_scores = pooled_scores[:, 0]
    music_scores = pooled_scores[:, 1]
    noise_scores = pooled_scores[:, 2]

    # Generate binary labels based on per-class thresholds.
    speech_mask = (speech_scores >= AST_SPEECH_THRESHOLD)
    music_mask = (music_scores >= AST_MUSIC_THRESHOLD)

    # Residual noise: only when neither speech nor music is active
    noise_mask = (~speech_mask & ~music_mask)


    batch[SPEECH] = speech_mask.int().tolist()
    batch[MUSIC] = music_mask.int().tolist()
    batch[NOISE] = noise_mask.int().tolist()
    batch[SPEECH_SCORE] = speech_scores.tolist()
    batch[MUSIC_SCORE] = music_scores.tolist()
    batch[NOISE_SCORE] = noise_scores.tolist()
    # All labels in this batch are from AST pseudo labeling.
    batch[LABEL_SOURCE_FIELD] = [LABEL_SOURCE_AST_PSEUDO] * len(segment_names)

    return batch


def main() -> None:

    with time_block("AST pseudo-labeling process"):
        settings = get_settings()
        metadata_dir: Path = settings.metadata_path
        segments_dir: Path = settings.segments_path

        base_manifest_path = metadata_dir / BLOCS_SMAD_V1
        logger.info(f'Loading base dataset from {base_manifest_path}...')
        ds: Dataset = Dataset.load_from_disk(base_manifest_path)

        # ds = ds.select(range(500))  # For testing with a smaller subset.

        map_fn = partial(_apply_ast_to_batch, segments_dir=segments_dir)

        logger.info('Applying AST teacher...')
        ds_ast = ds.map(
            map_fn,
            batched=True,
            batch_size=16,
            desc='AST pseudo labeling',
        )

        out_name = BLOCS_SMAD_V2_AST
        out_path = metadata_dir / out_name

        logger.info(f'Saving AST labeled dataset to {out_path}...')
        ds_ast.save_to_disk(out_path)
        ds_ast.to_csv(str(metadata_dir / f'{out_name}.csv'), index=False)
        # Log summary statistics of pseudo-labels.
        log_pseudo_label_stats(ds_ast, teacher_name="AST pseudo teacher")


if __name__ == '__main__':
    main()
