# models/smad_model.py

"""SMAD model definition using AST as the backbone.

This module defines a simple, reliable baseline for Speech/Music/Noise detection
using a pretrained Audio Spectrogram Transformer (AST) encoder with a lightweight
multi-label classification head.

Design goals:
- Keep the forward() contract simple: accept either raw waveforms or precomputed
  AST features.
- Prefer feature extraction in the dataloader/collate_fn for throughput, while
  still supporting ad-hoc waveform inference.
- Return raw logits (no sigmoid) so training code can choose an appropriate loss
  (e.g., BCEWithLogitsLoss) and thresholds.

Reference:
- Audio Spectrogram Transformer (AST): https://arxiv.org/abs/2104.01778
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from config import AST_MODEL_NAME, SAMPLE_RATE


# Keys commonly used by HF AST-like encoders. The dataset/collate can provide any
# subset of these depending on the chosen feature extractor / model variant.
AST_INPUT_KEYS = {
    'input_values',
    'attention_mask',
    'input_features',
    'pixel_values',
}


class SmadAstBaseline(nn.Module):
    """AST encoder with a lightweight multi-label head.

    Parameters
    ----------
    model_name:
        Hugging Face model id for an AST-like encoder.
    num_labels:
        Number of output labels (default: 3 for speech/music/noise).
    freeze_encoder:
        If True, disables gradient updates to the encoder for faster/cheaper
        training and to avoid overfitting on small labeled sets.
    dropout_p:
        Dropout probability applied before the classifier head.

    Notes
    -----
    - The forward() method expects a dict-like batch.
    - If waveform is provided, feature extraction is performed on-the-fly.
      For training, doing feature extraction in the dataloader is usually faster.
    """

    def __init__(
        self,
        model_name: str = AST_MODEL_NAME,
        num_labels: int = 3,
        freeze_encoder: bool = True,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)

        hidden_dim = int(config.hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        self._feature_extractor = None

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def _get_feature_extractor(self):
        """Lazily construct and cache the HF feature extractor for this model.

        We avoid instantiating the extractor in __init__ so importing/constructing
        the module remains lightweight when only precomputed features are used.
        """
        if self._feature_extractor is None:
            from transformers import AutoFeatureExtractor

            self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        return self._feature_extractor

    def _featurize_waveforms(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert a raw waveform batch into AST encoder inputs.

        Expects waveform shaped [B, T] or [B, 1, T] (mono). Returns a dictionary
        suitable for passing directly into a HF AST-like encoder.

        Note: this path is convenient for inference and debugging, but for training
        it's typically faster to precompute features in the dataloader/collate_fn.
        """
        # Allow callers to provide [B, 1, T] (common after loading mono audio).
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # HF feature extractors operate over lists of NumPy arrays.
        wave_list = [w.detach().cpu().numpy().astype(np.float32) for w in waveform]

        fe = self._get_feature_extractor()
        inputs = fe(
            wave_list,
            sampling_rate=getattr(fe, 'sampling_rate', SAMPLE_RATE),
            return_tensors='pt',
            padding=True,
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute logits for a batch.

        Supported batch formats
        ----------------------
        1) Raw audio
           - batch['waveform'] shaped [B, T] or [B, 1, T]
           - This method will run the HF feature extractor on-the-fly.

        2) Precomputed AST features
           - batch contains one or more keys in AST_INPUT_KEYS.
           - Values must be torch.Tensors (they will be moved onto the model device).

        Returns
        -------
        torch.Tensor
            Logits shaped [B, num_labels]. (Apply sigmoid externally if needed.)
        """
        # Prefer the waveform path only when explicitly provided.
        encoder_inputs: Optional[Dict[str, torch.Tensor]]

        if 'waveform' in batch:
            encoder_inputs = self._featurize_waveforms(batch['waveform'])
        else:
            encoder_inputs = {}
            for k in AST_INPUT_KEYS:
                v = batch.get(k, None)
                if isinstance(v, torch.Tensor):
                    encoder_inputs[k] = v.to(self.device)

            if not encoder_inputs:
                raise KeyError(
                    'No valid AST inputs found in batch. '
                    f'Expected one of: {sorted(AST_INPUT_KEYS)}. '
                    f'Got keys: {sorted(batch.keys())}'
                )

        # Encoder outputs include per-token hidden states. We follow the common
        # CLS-token pooling convention for classification.
        outputs = self.encoder(**encoder_inputs)
        hidden_states = outputs.last_hidden_state

        # Pool using the first token (CLS).
        pooled = hidden_states[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        return logits


def build_smad_model(
    device: torch.device,
    model_name: str = AST_MODEL_NAME,
    freeze_encoder: bool = True,
    dropout_p: float = 0.1,
) -> nn.Module:
    """Factory helper to build and place the SMAD baseline model on device."""
    model = SmadAstBaseline(
        model_name=model_name,
        num_labels=3,
        freeze_encoder=freeze_encoder,
        dropout_p=dropout_p,
    )
    model.to(device)
    return model
