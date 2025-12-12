from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
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
        segment_seconds: float = 10.0,
        split: Optional[str] = None,
        return_waveform: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.segments_dir = Path(segments_dir)
        self.sample_rate = sample_rate
        self.segment_seconds = float(segment_seconds)
        self.return_waveform = return_waveform
        df = pd.read_csv(self.manifest_path)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.mel = None
        if not self.return_waveform:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length, win_length=win_length
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        wav_path = self.segments_dir / row["segment_path"]
        waveform, sr = torchaudio.load(wav_path)
        if sr is None or not isinstance(sr, (int, float)) or sr <= 0:
            raise ValueError(f"Invalid sample rate {sr} (type: {type(sr)}) for file {wav_path}")
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=int(sr), new_freq=int(self.sample_rate))
            waveform = resampler(waveform)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Ensure a consistent segment length (spec: fixed-length windows).
        # Some segments at file boundaries may be shorter; pad with zeros.
        segment_seconds = float(getattr(self, "segment_seconds", 10.0))
        target_len = int(round(self.sample_rate * segment_seconds))
        if target_len > 0:
            cur_len = int(waveform.shape[1])
            if cur_len < target_len:
                waveform = F.pad(waveform, (0, target_len - cur_len))
            elif cur_len > target_len:
                waveform = waveform[:, :target_len]
        labels = torch.tensor([row[f"chosen_{c}_label"] for c in CLASSES], dtype=torch.float32)
        if self.return_waveform:
            # return waveform (1, time) for upstream feature extractor (e.g., AST)
            return {"waveform": waveform, "labels": labels}
        if self.mel is None:
            raise RuntimeError("MelSpectrogram transform is not initialized (return_waveform=True).")
        mel = self.mel(waveform)  # (1, n_mels, time)
        return {"mel": mel, "labels": labels}
