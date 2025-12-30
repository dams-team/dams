"""DAMS data processing package.

This package contains the complete data processing pipeline for Domain Adaptation
for Multi-Label Speech Activity Detection (DAMS), including manifest construction
and teacher model inference.

Subpackages:
    pipeline: Sequential manifest construction from raw audio to training data
    teachers: Pretrained teacher model inference for pseudo-labeling

The pipeline subpackage implements the core DAMS workflow:
    Raw audio → Segments → Acoustic stats → Teacher scores → Pseudo-labels → Training manifest

The teachers subpackage provides inference utilities for AudioSet-pretrained and
audio-text contrastive models used for pseudo-label generation.

For detailed usage, see the documentation in each subpackage.
"""

from . import pipeline
from . import teachers

__all__ = ["pipeline", "teachers"]
