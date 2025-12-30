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


CLASSES = ['speech', 'music', 'noise']


SegmentManifest = list['SegmentManifestRow']

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

# ======================================================================================
#  BLOCS SMAD Version & Artifacts (deprecated names for compatibility)
# ======================================================================================

# Versioned dataset folders (HuggingFace `save_to_disk()` outputs).
BLOCS_GOLD_INTERVALS = 'blocs_gold_intervals'
BLOCS_OVERLAP_MANIFEST = 'blocs_overlap_manifest'
BLOCS_SMAD_V1 = 'blocs_smad_v1'
BLOCS_SMAD_V2_AST = 'blocs_smad_v2_ast'
BLOCS_SMAD_V2_WHISPER = 'blocs_smad_v2_whisper'
BLOCS_SMAD_V2_CLAP = 'blocs_smad_v2_clap'
BLOCS_SMAD_V2_M2D = 'blocs_smad_v2_m2d'
BLOCS_SMAD_V2_PANNS = 'blocs_smad_v2_panns'
BLOCS_SMAD_V2_GOLD = 'blocs_smad_v2_gold'   # Gold labeled dataset.
BLOCS_SMAD_V3 = 'blocs_smad_v3'             # Fused teacher labels or first student pass.
BLOCS_SMAD_FINAL = 'blocs_smad_manifest_with_splits.csv'

# Base filenames for IRR and Gold annotation artifacts (Deprecated).
CSV_BLOCS_SMAD_SEGMENTS = 'blocs_smad_segments.csv'
CSV_BLOCS_SMAD_GOLD_ANNOTATIONS = 'gold_annotations.csv'

JSONL_BLOCS_SMAD_GOLD_ANNOTATIONS = 'blocs_smad_gold_annotations_v1.jsonl'
BLOCS_SMAD_GOLD_HF = 'blocs_smad_v2_gold'

BLOCS_SMAD_IRR_LOG = 'blocs_smad_irr_stats_v1.json'
BLOCS_SMAD_IRR_TABLE = 'blocs_smad_irr_pairs_v1.csv'


# ======================================================================================
# Required columns per pipeline stage
# ======================================================================================

REQUIRED_BASE_COLS = ['raw_file', 'segment_path', 'start_time', 'end_time']

REQUIRED_GOLD_COLS = [
    'segment_path',
    'raw_file',
    'is_irr_segment',
    'speech_gold',
    'music_gold',
    'noise_gold',
]

REQUIRED_PSEUDO_COLS = [
    'segment_path',
    'speech_pseudo',
    'music_pseudo',
    'noise_pseudo',
    'speech_score_fused',
    'music_score_fused',
    'noise_score_fused',
    'pseudo_label_source',
]

FINAL_LABEL_COLS = ['speech_label', 'music_label', 'noise_label']
FINAL_SCORE_COLS = ['speech_score', 'music_score', 'noise_score']


# ======================================================================================
#  Row and batch schemas
# ======================================================================================

class SegmentBaseRow(TypedDict):
    """Minimal base row for a single audio segment (v1/segments.csv)."""
    segment_path: str       # segment filename, e.g. 001_NO_RAD_0001_s0001.wav.
    raw_file: str           # original long form filename, e.g. 001_NO_RAD_0001.wav.

    start_time: float       # seconds from start of source file.
    end_time: float         # seconds from start of source file.


class SegmentManifestRow(TypedDict):
    """Full manifest row with labels and scores (v4/manifest.csv)."""
    segment_path: str    # segment filename, e.g. 001_NO_RAD_0001_s0001.wav
    raw_file: str        # original long form filename, e.g. 001_NO_RAD_0001.wav

    start_time: float    # seconds from start of source file.
    end_time: float      # seconds from start of source file.

    split: SplitName     # dev, test, unlabeled, unsplit
    label_source: LabelSource

    # Multi label targets (speech/music/noise) as 0/1 integers.
    speech_label: int
    music_label: int
    noise_label: int

    # Teacher scores per class (float or None).
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
