# utils/audio_io.py
"""
Audio input/output utility functions.

Provides functions to read and write audio files using torchaudio,
handling common settings such as sample rate and encoding.

Usage:
    from utils.audio_io import ...
"""

from pathlib import Path
from typing import Iterator, Tuple

import torch
import torchaudio

from config import SAMPLE_RATE, AUDIO_ENCODING, BITS_PER_SAMPLE


def load_waveform(path: Path) -> Tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate


def load_mono_resampled(
    path: Path,
    target_sr: int = SAMPLE_RATE,
) -> torch.Tensor:
    waveform, sr = load_waveform(path)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform


def save_waveform(
    path: Path,
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(
        str(path),
        waveform,
        sample_rate,
        encoding=AUDIO_ENCODING,
        bits_per_sample=BITS_PER_SAMPLE,
    )


def list_wavs(root: Path) -> Iterator[Path]:
    yield from root.rglob('*.wav')
