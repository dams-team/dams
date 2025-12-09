# data_processing/apply_m2d.py

"""
Apply M2D-CLAP zero-shot teacher model to generate pseudo-labels for audio segments.

Reads audio segments from disk, encodes them with a pretrained M2D-CLAP
model, scores them against speech / music / noise text prompts, and saves
resulting SMAD labels back to disk.

Usage:
    python -m data_processing.teachers.apple_m2d
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torchaudio
from datasets import Dataset

from config import get_settings, SAMPLE_RATE, M2D_CLAP_CHECKPOINT

from utils.dams_types import (
    BatchDict,
    SEGMENT_PATH,
    LABEL_SOURCE_FIELD,
    LABEL_SOURCE_M2D_ZS,
    SPEECH,
    MUSIC,
    NOISE,
    SPEECH_SCORE,
    MUSIC_SCORE,
    NOISE_SCORE,
    BLOCS_SMAD_V1,
    BLOCS_SMAD_V2_M2D,
)

from utils.logger import logger, log_pseudo_label_stats
from utils.timing import time_block

try:
    # First, try your vendored copy
    from utils.portable_m2d import PortableM2D
except ImportError:
    try:
        # Fallback to the original repo layout if someone has cloned it as-is
        from examples.portable_m2d import PortableM2D
    except ImportError as exc:  # pragma: no cover - helpful runtime error
        raise ImportError(
            "Could not import PortableM2D. Put `portable_m2d.py` in `utils/` "
            "or `examples/` so that `PortableM2D` is importable."
        ) from exc


# Simple cosine-similarity thresholds for mapping CLAP scores to SMAD labels.
# These are starting points; you will likely tune them using your validation
# set and gold-aligned segments. Originally 0.3 thresholds for each class.
M2D_CLAP_SPEECH_THRESHOLD: float = 0.24
M2D_CLAP_MUSIC_THRESHOLD: float = 0.26
M2D_CLAP_NOISE_THRESHOLD: float = 0.24


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device: torch.device = get_device()


def _build_m2d_clap_model(checkpoint_path: Path) -> PortableM2D:
    """Load a pretrained M2D-CLAP model and move it to the selected device."""

    logger.info(f"Loading M2D-CLAP teacher from {checkpoint_path}...")
    model = PortableM2D(str(checkpoint_path), flat_features=True)
    model.eval()
    model.to(device)
    return model


def _build_label_text_embeddings(model: PortableM2D) -> torch.Tensor:
    """Embed SMAD label prompts with the CLAP text encoder.

    Uses three compact prompts, one per label. You can change these prompts
    later if you want to probe more specific phrasing for speech / music /
    noise (for example multiple prompts per label averaged together).
    """

    label_prompts = [
        # Speech: clear, foreground voice, not just crowd murmur
        "Human speech: talking, conversation, announcer, DJ, interview, narration."
        " Includes breath sounds, laughter, sighs, coughs, lip noises. Clear vocal"
        " formants, not singing.",
        # Music: foreground musical content, speech only incidental
        "Music: instruments, melody, rhythm, harmony, beat patterns. Includes singing,"
        " rap, humming, whistling, and background music beds. Musical structure, not"
        " ordinary speech.",
        # Noise: no clear speech or melodic music at all
        "Noise: environmental and mechanical sound such as static, rumble, wind,"
        " traffic, crowd murmur, microphone hiss. Background sound without clear words"
        " or melody.",
    ]

    with torch.no_grad():
        text_embs = model.encode_clap_text(label_prompts, truncate=True)
        text_embs = torch.nn.functional.normalize(text_embs, p=2, dim=-1)

    return text_embs.to(device)


# Instantiate global teacher and label prototypes once so that `map` calls
# only handle audio I/O and small matrix multiplies.
settings = get_settings()
_m2d_clap_model: PortableM2D = _build_m2d_clap_model(M2D_CLAP_CHECKPOINT)
_LABEL_TEXT_EMBEDDINGS: torch.Tensor = _build_label_text_embeddings(_m2d_clap_model)


def _load_segment(segments_dir: Path, segment_name: str) -> torch.Tensor:
    seg_path = segments_dir / segment_name
    waveform, sr = torchaudio.load(seg_path)

    assert sr == SAMPLE_RATE, f"Unexpected sample rate for {segment_name}: {sr}"
    assert waveform.size(0) == 1, f"Expected mono: found {waveform.size(0)} channels"

    target_len = SAMPLE_RATE * 10
    num_samples = waveform.size(1)
    if num_samples < target_len:  # pad with zeros if shorter than 10s.
        pad = target_len - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif num_samples > target_len:  # truncate if longer than 10s.
        waveform = waveform[:, :target_len]

    return waveform.squeeze(0)  # 1D tensor, shape [samples]


def _apply_m2d_clap_to_batch(batch: BatchDict, segments_dir: Path) -> BatchDict:
    segment_names: list[str] = batch[SEGMENT_PATH]

    waveforms = [_load_segment(segments_dir, name) for name in segment_names]
    batch_audio = torch.stack(waveforms, dim=0)  # [batch, samples]
    batch_audio = batch_audio.to(device)

    with torch.no_grad():
        # Encode audio with the CLAP audio head.
        audio_embs = _m2d_clap_model.encode_clap_audio(batch_audio)
        audio_embs = torch.nn.functional.normalize(audio_embs, p=2, dim=-1)

        # Cosine similarity between audio embeddings and label text embeddings.
        # Shape: [batch, 3] corresponding to [speech, music, noise].
        scores = audio_embs @ _LABEL_TEXT_EMBEDDINGS.T

    scores_cpu = scores.cpu()
    speech_scores = scores_cpu[:, 0]
    music_scores = scores_cpu[:, 1]
    noise_scores = scores_cpu[:, 2]

    # Binary masks based on per-class thresholds. This is genuinely multilabel
    # (segments can be tagged as both speech and music, etc.).
    speech_mask = speech_scores >= M2D_CLAP_SPEECH_THRESHOLD
    music_mask = music_scores >= M2D_CLAP_MUSIC_THRESHOLD
    noise_mask = noise_scores >= M2D_CLAP_NOISE_THRESHOLD

    batch[SPEECH] = speech_mask.int().tolist()
    batch[MUSIC] = music_mask.int().tolist()
    batch[NOISE] = noise_mask.int().tolist()
    batch[SPEECH_SCORE] = speech_scores.tolist()
    batch[MUSIC_SCORE] = music_scores.tolist()
    batch[NOISE_SCORE] = noise_scores.tolist()

    # Optional: keep the raw similarity scores for analysis.
    batch["m2d_clap_scores"] = scores_cpu.tolist()

    # All labels in this batch come from the M2D-CLAP zero-shot teacher.
    batch[LABEL_SOURCE_FIELD] = [LABEL_SOURCE_M2D_ZS] * len(segment_names)

    return batch


def main() -> None:

    with time_block("M2D-CLAP zero-shot pseudo-labeling process"):
        metadata_dir: Path = settings.metadata_path
        segments_dir: Path = settings.segments_path

        base_manifest_path = metadata_dir / BLOCS_SMAD_V1
        logger.info(f"Loading base dataset from {base_manifest_path}...")
        ds: Dataset = Dataset.load_from_disk(base_manifest_path)

        # ds = ds.select(range(500))  # For debugging with a smaller subset.

        map_fn = partial(_apply_m2d_clap_to_batch, segments_dir=segments_dir)

        logger.info("Applying M2D-CLAP zero-shot teacher...")
        ds_m2d = ds.map(
            map_fn,
            batched=True,
            batch_size=16,
            desc="M2D-CLAP zero-shot labeling",
        )

        out_name = BLOCS_SMAD_V2_M2D
        out_path = metadata_dir / out_name

        logger.info(f"Saving M2D-CLAP labeled dataset to {out_path}...")
        ds_m2d.save_to_disk(out_path)
        ds_m2d.to_csv(str(metadata_dir / f"{out_name}.csv"), index=False)

        # Log summary statistics of pseudo-labels.
        log_pseudo_label_stats(ds_m2d, teacher_name="M2D-CLAP zero-shot teacher")

        logger.info(
            f"M2D-CLAP thresholds: "
            f"speech_threshold={M2D_CLAP_SPEECH_THRESHOLD:.3f}, "
            f"music_threshold={M2D_CLAP_MUSIC_THRESHOLD:.3f}, "
            f"noise_threshold={M2D_CLAP_NOISE_THRESHOLD:.3f}"
        )


if __name__ == "__main__":
    main()
