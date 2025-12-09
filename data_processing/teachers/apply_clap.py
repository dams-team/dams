# data_processing/teachers/apply_clap.py

from pathlib import Path
import torch
import torchaudio
import pandas as pd
from datasets import Dataset
from transformers import pipeline
import torch.nn.functional as F

from config import get_settings, CLAP_MODEL_NAME
from utils.dams_types import (
    BatchDict,
    SEGMENT_PATH,
    SPEECH, MUSIC, NOISE,
    SPEECH_SCORE, MUSIC_SCORE, NOISE_SCORE,
    LABEL_SOURCE_FIELD, LABEL_SOURCE_CLAP_ZS,
    BLOCS_SMAD_V1, BLOCS_SMAD_V2_CLAP
)

from utils.logger import logger, log_pseudo_label_stats
from utils.timing import time_block

CLAP_SAMPLE_RATE = 48_000

# Thresholds for turning CLAP scores into multi-label decisions
CLAP_SPEECH_THRESH = 0.45
CLAP_MUSIC_THRESH = 0.35
CLAP_NOISE_THRESH = 0.35

# Zero-shot pipeline for CLAP (MPS if available, otherwise CPU)
if torch.backends.mps.is_available():
    _clap_device = "mps"
    device_arg = 0  # pipeline uses 0 for the first GPU-like device
else:
    _clap_device = "cpu"
    device_arg = -1

logger.info(f"Initializing zero-shot audio pipeline with {CLAP_MODEL_NAME} on {_clap_device}...")
_CLAP_CANDIDATE_LABELS = [
    # Speech: clear, foreground voice, not just crowd murmur
    "An audio recording where clear spoken human voice is the main sound in the"
    " foreground (talking, announcements, conversation or monologue)."
    " Any music or background noise is quiet and not the focus.",
    # Music: foreground musical content, speech only incidental
    "An audio recording where music is the main sound in the foreground (instruments and"
    " or singing with melody or rhythm). Any speech is brief or in the background and"
    " not the focus.",
    # Noise: no clear speech or melodic music at all
    "An audio recording that contains environmental or background noise only"
    " (traffic, nature, machinery, room tone, crowd hubbub) with no clearly intelligible"
    " speech and no clear melodic music.",
]
_CLAP_PIPE = pipeline(
    task="zero-shot-audio-classification",
    model=CLAP_MODEL_NAME,
    device=device_arg,
    top_k=None,  # Return scores for all candidate labels
    multi_label=True,
)

def _apply_clap_to_batch(batch: BatchDict, segments_dir: Path) -> BatchDict:
    names = batch[SEGMENT_PATH]

    s_scores, m_scores, n_scores = [], [], []
    s_labels, m_labels, n_labels = [], [], []

    for name in names:
        path = segments_dir / name
        waveform, sr = torchaudio.load(path)

        # Resample to CLAP expected rate
        if sr != CLAP_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=CLAP_SAMPLE_RATE,
            )
            sr = CLAP_SAMPLE_RATE

        # Pipeline expects a 1D numpy array for audio
        audio_np = waveform.squeeze(0).numpy()

        # Run zero-shot classification with our three candidate labels
        outputs = _CLAP_PIPE(
            audio_np,
            candidate_labels=_CLAP_CANDIDATE_LABELS,
        )
        # `outputs` is a list of dicts like [{"label": ..., "score": ...}, ...]
        # Map back to our three labels in the original order
        label_to_score = {o["label"]: float(o["score"]) for o in outputs}
        s = label_to_score[_CLAP_CANDIDATE_LABELS[0]]
        m = label_to_score[_CLAP_CANDIDATE_LABELS[1]]
        n = label_to_score[_CLAP_CANDIDATE_LABELS[2]]

        s_scores.append(s)
        m_scores.append(m)
        n_scores.append(n)

        # Multi-label decision: allow speech, music, and noise to be 1 independently
        s_labels.append(int(s >= CLAP_SPEECH_THRESH))
        m_labels.append(int(m >= CLAP_MUSIC_THRESH))
        n_labels.append(int(n >= CLAP_NOISE_THRESH))

    batch[SPEECH_SCORE], batch[MUSIC_SCORE], batch[NOISE_SCORE] = s_scores, m_scores, n_scores
    batch[SPEECH], batch[MUSIC], batch[NOISE] = s_labels, m_labels, n_labels
    batch[LABEL_SOURCE_FIELD] = [LABEL_SOURCE_CLAP_ZS] * len(names)

    return batch


def main() -> None:

    with time_block("CLAP teacher runetime"):

        settings = get_settings()
        metadata_dir = settings.metadata_path
        segments_dir = settings.segments_path

        base = metadata_dir / BLOCS_SMAD_V1
        logger.info(f"Loading {base}...")
        ds = Dataset.load_from_disk(base)

        # If this dataset already has generic score/label columns from another teacher,
        # drop them so we can write the CLAP scores/labels cleanly for this version.
        cols_to_drop = [
            SPEECH_SCORE,
            MUSIC_SCORE,
            NOISE_SCORE,
            SPEECH,
            MUSIC,
            NOISE,
            LABEL_SOURCE_FIELD,
        ]
        existing = [c for c in cols_to_drop if c in ds.column_names]
        if existing:
            # logger.info(f"Removing existing temp columns before adding CLAP columns: "
            #            f"{existing}")
            ds = ds.remove_columns(existing)

        logger.info("Applying CLAP teacher (sequential loop over segments)...")

        n = len(ds)
        speech_scores_all = []
        music_scores_all = []
        noise_scores_all = []
        speech_labels_all = []
        music_labels_all = []
        noise_labels_all = []
        label_sources_all = []

        for idx in range(n):
            # Take a single-example batch from the dataset
            batch = ds[idx:idx + 1]  # dict of lists, size 1

            batch = _apply_clap_to_batch(batch, segments_dir)

            speech_scores_all.extend(batch[SPEECH_SCORE])
            music_scores_all.extend(batch[MUSIC_SCORE])
            noise_scores_all.extend(batch[NOISE_SCORE])
            speech_labels_all.extend(batch[SPEECH])
            music_labels_all.extend(batch[MUSIC])
            noise_labels_all.extend(batch[NOISE])
            label_sources_all.extend(batch[LABEL_SOURCE_FIELD])

          #  if (idx + 1) % 100 == 0 or idx + 1 == n:
          #      logger.info(f'  Processed {idx + 1}/{n} segments with CLAP')

        ds_clap = ds.add_column(SPEECH_SCORE, speech_scores_all)
        ds_clap = ds_clap.add_column(MUSIC_SCORE, music_scores_all)
        ds_clap = ds_clap.add_column(NOISE_SCORE, noise_scores_all)
        ds_clap = ds_clap.add_column(SPEECH, speech_labels_all)
        ds_clap = ds_clap.add_column(MUSIC, music_labels_all)
        ds_clap = ds_clap.add_column(NOISE, noise_labels_all)
        ds_clap = ds_clap.add_column(LABEL_SOURCE_FIELD, label_sources_all)

        out = metadata_dir / BLOCS_SMAD_V2_CLAP
        logger.info(f"Saving to {out}...")
        ds_clap.save_to_disk(out)
        ds_clap.to_csv(str(out) + ".csv", index=False)

        log_pseudo_label_stats(ds_clap, teacher_name="CLAP zero-shot teacher")

if __name__ == "__main__":
    main()
