# utils/dams_types.py

"""
Shared types and field names for DAMS BLOCS segment manifests and
batch dictionaries.

This module is the single source of truth for:
  - split and label enums
  - manifest field name constants
  - row and batch schemas
  - dataset version identifiers

  Usage:
        from utils.dams_types import ...
"""

from typing import TypedDict, Literal

SegmentManifest = list['SegmentRow']

# ================================
#  Split and label enums
# ================================

LABEL_SOURCE_NONE = 'none'
LABEL_SOURCE_GOLD = 'gold'
LABEL_SOURCE_AST_PSEUDO = 'ast_pseudo'
LABEL_SOURCE_WHISPER_PSEUDO = 'whisper_pseudo'
LABEL_SOURCE_CLAP_ZS = 'clap_zero_shot'
LABEL_SOURCE_M2D_ZS = 'm2d_zero_shot'
LABEL_SOURCE_PANNS_PSEUDO = 'panns_pseudo'

# Splits used in the manifest.
SplitName = Literal['train', 'dev', 'test', 'unlabeled', 'unsplit']

# Where a given label came from.
LabelSource = Literal[
    'none',           # no label yet
    'gold',           # human labeled
    'ast_pseudo',     # AST teacher pseudo label
    'whisper_pseudo', # Whisper AT teacher pseudo label
    'panns_pseudo',   # PANNs teacher pseudo label
    'clap_zero_shot', # CLAP teacher zero-shot label
    'm2d_zero_shot',  # M2D-CLAP teacher zero-shot label
]

# ================================
#  Manifest field name constants
# ================================

# Core identification and timing fields.
RAW_FILE = 'raw_file'
SEGMENT_PATH = 'segment_path'
START_TIME = 'start_time'
END_TIME = 'end_time'
SPLIT = 'split'
LABEL_SOURCE_FIELD = 'label_source'

# Multi-label targets.
SPEECH = 'speech_label'
MUSIC = 'music_label'
NOISE = 'noise_label'

# Optional scores from teachers, same order as above.
SPEECH_SCORE = 'speech_score'
MUSIC_SCORE = 'music_score'
NOISE_SCORE = 'noise_score'

# Optional review flag for human annotation.
NEEDS_REVIEW = 'needs_review'

# ================================
#  BLOCS SMAD Version & Artifacts
# ================================

# Versioned dataset folders (HuggingFace `save_to_disk()` outputs).
BLOCS_GOLD_INTERVALS = 'blocs_gold_intervals'
BLOCS_OVERLAP_MANIFEST = 'blocs_overlap_manifest'
BLOCS_SMAD_SEGMENTS = 'blocs_smad_segments'
BLOCS_SMAD_V1 = 'blocs_smad_v1'
BLOCS_SMAD_V2_AST = 'blocs_smad_v2_ast'
BLOCS_SMAD_V2_WHISPER = 'blocs_smad_v2_whisper'
BLOCS_SMAD_V2_CLAP = 'blocs_smad_v2_clap'
BLOCS_SMAD_V2_M2D = 'blocs_smad_v2_m2d'
BLOCS_SMAD_V2_PANNS = 'blocs_smad_v2_panns'
BLOCS_SMAD_V2_GOLD = 'blocs_smad_v2_gold'   # Gold labeled dataset.
BLOCS_SMAD_V3 = 'blocs_smad_v3'             # Fused teacher labels or first student pass.
BLOCS_SMAD_FINAL = 'blocs_smad_final'       # Final dataset for training student/benchmarks.

# Base CSV artifact filenames
CSV_BLOCS_SMAD_SEGMENTS = 'blocs_smad_segments.csv'
CSV_BLOCS_SMAD_LABELS = 'blocs_smad_labels.csv'
CSV_BLOCS_OVERLAP_MANIFEST = 'blocs_overlap_manifest.csv'
CSV_BLOCS_SMAD_GOLD_ANNOTATIONS = 'blocs_smad_gold_annotations_v1.csv'

# Gold annotation artifacts
CSV_BLOCS_SMAD_GOLD_LABELS = 'blocs_smad_labels_gold_v1.csv'  # Trains student model.
JSONL_BLOCS_SMAD_GOLD_ANNOTATIONS = 'blocs_smad_gold_annotations_v1.jsonl'
BLOCS_SMAD_GOLD_HF = 'blocs_smad_v2_gold'

# IRR artifacts
BLOCS_SMAD_IRR_LOG = 'blocs_smad_irr_stats_v1.json'
BLOCS_SMAD_IRR_TABLE = 'blocs_smad_irr_pairs_v1.csv'

# ================================
#  Row and batch schemas
# ================================

class SegmentRow(TypedDict):
    """
    One row in the BLOCS segment manifest.

    This matches what ends up in blocs_smad_* metadata,
    regardless of whether labels are gold or pseudo.
    """
    raw_file: str        # original long form filename, e.g. 001_NO_RAD_0001.wav
    segment_path: str    # segment filename, e.g. 001_NO_RAD_0001_s0001.wav

    start_time: float    # seconds from start of raw file
    end_time: float      # seconds from start of raw file

    split: SplitName     # dev, test, unlabeled, unsplit
    label_source: LabelSource

    # Multi label targets.
    speech_label: int
    music_label: int
    noise_label: int

    # Optional scores from teachers, same order as above.
    speech_score: float | None
    music_score: float | None
    noise_score: float | None

    # Annotation Review flag for gold labeling.
    needs_review: int | None  # 0 or 1


class BatchDict(TypedDict, total=False):
    """
    A batch dictionary used in data processing functions
    (e.g., Dataset.map callbacks).

    Keys line up with SegmentRow fields, but values are
    batched into lists.
    """
    segment_path: list[str]

    # Multi label targets (0/1) per class.
    speech_label: list[int]
    music_label: list[int]
    noise_label: list[int]
    # Teacher scores per class.
    speech_score: list[float | None]
    music_score: list[float | None]
    noise_score: list[float | None]

    label_source: list[LabelSource]     # See LabelSource above.
    ast_probs: list[list[float]]        # For the AST teacher.
    whisper_probs: list[list[float]]    # For the Whisper AT teacher.
