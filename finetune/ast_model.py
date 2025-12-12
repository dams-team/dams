from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification


class ASTClassifier(torch.nn.Module):
    """
    Wrapper around a pretrained AST audio classifier from Hugging Face.
    Expects input_values (batch of waveforms) already padded/stacked by the feature extractor.
    """

    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", num_labels: int = 3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        # Allow the classification head to be resized from the checkpoint (527) to our label count.
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name, config=self.config, ignore_mismatched_sizes=True
        )

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.model(input_values=input_values, attention_mask=attention_mask)
        return out.logits


def get_feature_extractor(model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593") -> AutoFeatureExtractor:
    return AutoFeatureExtractor.from_pretrained(model_name)
