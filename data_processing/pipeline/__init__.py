"""Manifest construction pipeline for DAMS.

This pipeline progressively builds manifests from raw audio to training-ready data
through a series of enrichment stages. Each stage adds features, scores, labels, or
split assignments to produce the final training manifest.

Pipeline stages (in order):
1. segment_audio: Segment long-form recordings into fixed-length overlapping windows
2. compute_acoustic_stats: Compute acoustic features (RMS, ZCR, SNR) and QC flags
3. merge_teacher_scores: Merge per-teacher predictions into unified score table
4. fuse_pseudo_labels: Fuse teacher scores into calibrated pseudo-labels via voting
5. assign_splits: Assign train/dev/test splits with leakage controls, emit final manifest

Each module is designed to be run independently as a CLI script:
    python -m data_processing.pipeline.segment_audio --corpus blocs
    python -m data_processing.pipeline.compute_acoustic_stats
    python -m data_processing.pipeline.merge_teacher_scores
    python -m data_processing.pipeline.fuse_pseudo_labels
    python -m data_processing.pipeline.assign_splits [args]

See paper Section 3 and Algorithm 1 for technical details on the DAMS pipeline.
"""

__all__ = [
    "segment_audio",
    "compute_acoustic_stats",
    "merge_teacher_scores",
    "fuse_pseudo_labels",
    "assign_splits",
]
