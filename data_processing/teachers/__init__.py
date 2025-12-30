"""Pretrained teacher models for DAMS pseudo-labeling.

This subpackage provides inference utilities for AudioSet-pretrained and audio-text
contrastive models used to generate pseudo-labels for Speech-Music-Noise Activity
Detection (SMAD). Each teacher produces per-segment scores that are later calibrated
and fused in the manifest pipeline.

Available teachers:
    apply_ast: Audio Spectrogram Transformer (AudioSet-pretrained, supervised)
    apply_whisper: Whisper-AT audio tagging head (ASR-pretrained, repurposed for tagging)
    apply_clap: LAION-CLAP audio-text contrastive model (zero-shot prompting)
    apply_m2d: M2D-CLAP masked modeling + contrastive alignment (zero-shot prompting)

Teacher outputs:
    Each teacher produces per-segment scores for speech, music, and noise in [0, 1],
    along with binary pseudo-labels based on calibrated thresholds. These scores are
    later merged (merge_teacher_scores.py) and fused (fuse_pseudo_labels.py) to
    produce the final pseudo-label supervision signal.

Usage:
    Run each teacher independently to generate per-teacher labeled datasets:
        python -m data_processing.teachers.apply_ast
        python -m data_processing.teachers.apply_whisper
        python -m data_processing.teachers.apply_clap
        python -m data_processing.teachers.apply_m2d

See paper Section 3.4 for technical details on teacher architectures and selection.
"""

__all__ = [
    "apply_ast",
    "apply_whisper",
    "apply_clap",
    "apply_m2d",
]
