from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

CLASSES = ["speech", "music", "noise"]


class SmadDataset(Dataset):
    """
    Simple dataset for finetuning:
    - expects a manifest with columns: segment_path, chosen_*_label, optional split
    - loads WAV from segments_dir / segment_path
    - computes Mel spectrogram
    """

    def __init__(
        self,
        manifest_path: Path,
        segments_dir: Path,
        sample_rate: int = 16000,
        n_mels: int = 128,
        hop_length: int = 160,
        win_length: int = 400,
        split: Optional[str] = None,
        return_waveform: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.segments_dir = Path(segments_dir)
        self.sample_rate = sample_rate
        self.return_waveform = return_waveform
        df = pd.read_csv(self.manifest_path)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length, win_length=win_length
        )
        self.resampler = torchaudio.transforms.Resample(orig_freq=None, new_freq=sample_rate)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        wav_path = self.segments_dir / row["segment_path"]
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = self.resampler(waveform)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        labels = torch.tensor([row[f"chosen_{c}_label"] for c in CLASSES], dtype=torch.float32)
        if self.return_waveform:
            # return waveform (1, time) for upstream feature extractor (e.g., AST)
            return {"waveform": waveform, "labels": labels}
        mel = self.mel(waveform)  # (1, n_mels, time)
        return {"mel": mel, "labels": labels}
