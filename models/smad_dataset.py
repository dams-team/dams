# models/smad_dataset.py

"""PyTorch dataset + dataloader helpers for BLOCS SMAD.

This module wraps a BLOCS SMAD manifest (CSV/Parquet or HF save_to_disk() folder)
into a PyTorch Dataset that yields:
- waveform: mono audio shaped [1, T]
- labels: float32 multi-hot vector ordered as [speech, music, noise]

Example:
    from models.smad_dataset import make_smad_dataloaders, build_ast_collate_fn

    collate_fn = build_ast_collate_fn(model_name=cfg.model_name)

    train_loader, dev_loader, test_loader = make_smad_dataloaders(
        manifest_name=cfg.manifest_name,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        train_mode='gold_only',   # or 'pseudo_only' or 'all'
)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset

from config import AST_MODEL_NAME, SAMPLE_RATE, SEGMENT_LEN, get_settings
from utils.audio_io import load_mono_resampled
from utils.dams_types import BLOCS_SMAD_FINAL, MUSIC, NOISE, SEGMENT_PATH, SPEECH, SplitName


TrainMode = Literal['all', 'gold_only', 'pseudo_only']


@dataclass
class SmadDatasetConfig:
    """Configuration options for the SmadDataset class.

    'manifest_name' may be either:
    - an absolute/relative path to an existing manifest file or HF dataset folder, or
    - a name resolved under 'Settings.metadata_path'.
    """

    manifest_name: str = BLOCS_SMAD_FINAL
    split: str = 'train'
    sample_rate: int = SAMPLE_RATE
    max_duration_sec: float = SEGMENT_LEN
    mono: bool = True
    # Only affects split=='train' when the manifest includes an is_gold column.
    # - 'all': use both gold and pseudo labels
    # - 'gold_only': train only on gold labeled rows
    # - 'pseudo_only': train only on pseudo labeled rows
    train_mode: TrainMode = 'all'


class SmadDataset(Dataset):
    """PyTorch Dataset for BLOCS SMAD segments.

    Each item yields a dict with:
    - waveform: torch.Tensor shaped [1, T] (mono)
    - labels: torch.Tensor shaped [3] ordered [speech, music, noise]
    - segment_path: str (relative path under segments_root)
    - is_gold: torch.BoolTensor scalar (if column missing, defaults False)

    This class intentionally performs minimal preprocessing (pad/trim to a fixed
    duration). Feature extraction can be done either:
    - in the model (waveform -> HF feature extractor), or
    - in the dataloader via AstFeatureCollator (often faster).
    """

    def __init__(self, config: SmadDatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.settings = get_settings()
        self.segments_root = self.settings.segments_path

        # Allow callers to pass either an explicit path or a metadata-relative name.
        manifest_candidate = Path(config.manifest_name)
        if manifest_candidate.exists():
            self.manifest_path = manifest_candidate
        else:
            self.manifest_path = self.settings.metadata_path / config.manifest_name

        dataset = self._load_manifest(self.manifest_path)

        # Filter by split if present.
        if 'split' in dataset.column_names:
            dataset = dataset.filter(lambda row: row['split'] == config.split)

        # Optional train-only filtering to support baselines without regenerating manifests.
        # IMPORTANT: on this project, gold rows also have teacher scores, so has_pseudo==True
        # does NOT imply pseudo-only. For pseudo-only, we must explicitly exclude gold.
        if config.split == 'train':
            mode = str(getattr(config, 'train_mode', 'all'))
            if mode not in ['all', 'gold_only', 'pseudo_only']:
                raise ValueError(f'Invalid train_mode: {mode}')

            if mode != 'all' and 'is_gold' not in dataset.column_names:
                raise KeyError("train_mode requires an 'is_gold' column in the manifest")

            if mode == 'gold_only':
                dataset = dataset.filter(lambda row: bool(row.get('is_gold', False)))

            elif mode == 'pseudo_only':
                # Exclude any gold rows first.
                dataset = dataset.filter(lambda row: not bool(row.get('is_gold', False)))
                # If the manifest has has_pseudo, enforce it so we do not pull in truly-unlabeled rows.
                if 'has_pseudo' in dataset.column_names:
                    dataset = dataset.filter(lambda row: bool(row.get('has_pseudo', False)))

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_manifest(self, path: Path) -> HFDataset:
        """Load a manifest either from a HF dataset directory or a flat file.

        Supported inputs:
        - A directory created by datasets.Dataset.save_to_disk()
        - A .csv or .parquet file
        """

        if path.is_dir():
            return load_from_disk(str(path))

        if not path.exists():
            raise FileNotFoundError(f'Manifest not found: {path}')

        suffix = path.suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(path)
            return HFDataset.from_pandas(df, preserve_index=False)

        if suffix == '.parquet':
            df = pd.read_parquet(path)
            return HFDataset.from_pandas(df, preserve_index=False)

        raise ValueError(f'Unsupported manifest format: {path} (expected a directory,'
                         f' .csv, or .parquet)')

    def _load_waveform(self, segment_path: str) -> torch.Tensor:
        """Load, resample, and pad/trim a segment waveform.

        Returns a mono tensor shaped [1, T] where T corresponds to
        sample_rate * max_duration_sec.
        """

        # Segment paths in the manifest are stored relative to the segments root.
        full_path = self.segments_root / segment_path
        waveform = load_mono_resampled(
            full_path,
            target_sr=self.config.sample_rate,
        )

        target_num_samples = int(self.config.sample_rate * self.config.max_duration_sec)
        num_samples = waveform.shape[-1]  # [1, T]
        # Enforce fixed-length segments for batching.
        if num_samples > target_num_samples:
            waveform = waveform[..., :target_num_samples]
        elif num_samples < target_num_samples:
            pad_amount = target_num_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Fetch one example.

        Returns a dict containing waveform + labels, plus lightweight metadata used
        for debugging and mixed-supervision weighting.
        """
        row = self.dataset[index]
        segment_path = row[SEGMENT_PATH]
        waveform = self._load_waveform(segment_path)
        # Labels are stored as float indicators and ordered consistently.
        label_vector = torch.tensor(
            [
                float(row[SPEECH]),
                float(row[MUSIC]),
                float(row[NOISE]),
            ],
            dtype=torch.float32,
        )

        # Mixed supervision: expose whether this row is gold so the trainer can weight loss.
        # Default to False when the column is missing.
        is_gold = bool(row.get('is_gold', False))

        example: Dict[str, Any] = {
            'waveform': waveform,  # [1, T]
            'labels': label_vector,  # [3]
            'segment_path': segment_path,
            'is_gold': torch.tensor(is_gold, dtype=torch.bool),
            'label_source': row.get('label_source', None),
        }
        return example


class AstFeatureCollator:
    """Pickle safe AST collator for DataLoader workers on macOS.

    DataLoader uses spawn, so the collate callable must be pickleable.
    Feature extractor is lazily created inside each worker.
    """

    def __init__(self, model_name: str, sample_rate: int) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self._fe = None

    def __getstate__(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'sample_rate': self.sample_rate,
            '_fe': None,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.model_name = state['model_name']
        self.sample_rate = state['sample_rate']
        self._fe = None

    def _get_fe(self):
        """Lazily load the HF feature extractor inside each worker process."""
        if self._fe is None:
            from transformers import AutoFeatureExtractor

            self._fe = AutoFeatureExtractor.from_pretrained(self.model_name)
        return self._fe

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of examples and compute AST inputs.

        Produces a batch dict compatible with AST-like HF encoders and preserves
        labels + is_gold for training.
        """
        fe = self._get_fe()

        waves = [ex['waveform'].squeeze(0) for ex in batch]
        labels = torch.stack([ex['labels'] for ex in batch], dim=0)
        is_gold = torch.stack(
            [ex.get('is_gold', torch.tensor(False, dtype=torch.bool)) for ex in batch],
            dim=0,
        )
        segment_paths = [ex['segment_path'] for ex in batch]
        label_sources = [ex.get('label_source', None) for ex in batch]
        # Convert to NumPy for the HF feature extractor API.
        wave_list = [w.detach().cpu().numpy().astype(np.float32) for w in waves]
        inputs = fe(
            wave_list,
            sampling_rate=self.sample_rate,
            return_tensors='pt',
            padding=True,
        )

        # Normalize key name across Transformers versions.
        if 'input_values' not in inputs and 'input_features' in inputs:
            inputs['input_values'] = inputs.pop('input_features')

        out: Dict[str, Any] = {**inputs}
        out['labels'] = labels
        out['is_gold'] = is_gold
        out['segment_path'] = segment_paths
        out['label_source'] = label_sources
        return out


def build_ast_collate_fn(
    model_name: str = AST_MODEL_NAME,
    sample_rate: int = SAMPLE_RATE,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    return AstFeatureCollator(model_name=model_name, sample_rate=sample_rate)


def make_smad_dataloaders(
    manifest_name: str = BLOCS_SMAD_FINAL,
    batch_size: int = 8,
    num_workers: int = 4,
    collate_fn: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = None,
    train_mode: TrainMode = 'all',
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Construct DataLoaders for train, dev, and test splits."""

    def _build_loader(split: SplitName, shuffle: bool) -> Optional[DataLoader]:
        cfg = SmadDatasetConfig(
            manifest_name=manifest_name,
            split=split,
        )
        if split == 'train':
            cfg.train_mode = train_mode

        ds = SmadDataset(cfg)
        if len(ds) == 0:
            return None

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )

    train_loader = _build_loader('train', shuffle=True)
    dev_loader = _build_loader('dev', shuffle=False)
    test_loader = _build_loader('test', shuffle=False)

    return train_loader, dev_loader, test_loader
