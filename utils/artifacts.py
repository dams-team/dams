# utils/artifacts.py

"""Artifact filenames used across the DAMS pipeline.

These are corpus-agnostic base filenames. Versioning is handled via dated or
numbered subdirectories under data/metadata/ (i.e., v1/, v2/, v3/, v4/) to track
experimental runs and avoid overwriting intermediate results.

Directory structure:
    data/metadata/blocs_smad/
        ├── v1/
        │   ├── segments.csv
        │   └── qc_audio_stats.csv
        ├── v2/
        │   ├── manifest.csv
        │   ├── manifest/                 # HF dataset
        │   └── teachers/
        │       ├── {ast,clap,m2d,whisper}.csv
        │       └── {ast,clap,m2d,whisper}/   # HF datasets
        ├── v3/
        │   ├── manifest.csv
        │   ├── manifest/
        │   ├── fusion_policy.json
        │   └── thresholds.csv
        └── v4/
            ├── manifest.csv
            └── manifest/
"""

# ======================================================================================
# Version directory names
# ======================================================================================

V1  = 'v1'  # Segmentation + acoustic stats
V2  = 'v2'  # Teacher predictions
V3  = 'v3'  # Fused pseudo-labels
V4  = 'v4'  # Train/dev/test splits

# ======================================================================================
# Version-specific auxiliary files
# ======================================================================================

# v1: Segmentation + Gold annotations + optional acoustic stats.
SEGMENTS_CSV = 'segments.csv'
GOLD_ANNOTATIONS_CSV = 'gold_annotations.csv'
QC_AUDIO_STATS_CSV = 'qc_audio_stats.csv'

# v2+: manifest (CSV and HF dataset share base name)
MANIFEST = 'manifest'  # .csv for file, / for HF dir

# v2: Teacher predictions (Consolidated per-teacher manifest).
TEACHERS_DIR = 'teachers'  # Directory containing per-teacher manifests.

# v3: Fusion artifacts.
FUSION_POLICY_JSON = 'fusion_policy.json'
FUSION_THRESHOLDS_CSV = 'thresholds.csv'

# v4: Splits.
SPLIT_MAP_CSV = 'split_map.csv'     # Optional mapping of files to splits.
SPLIT_LOG_JSON = 'split_log.json'   # Summary of split statistics.
