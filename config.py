# config.py

"""Central configuration module for the DAMS audio processing project.

This module defines global constants, model identifiers, and filesystem paths
used across data processing, teacher inference, and training components. It
also provides the `Settings` class, which loads environment-dependent values
(such as Backblaze B2 credentials and data directories) from a `.env` file or
system environment variables. The `get_settings()` function exposes a cached
singleton instance of these settings for use throughout the codebase.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_PATH = PROJECT_ROOT / 'checkpoints' # Directories for non-HF checkpoints.

# ================================
# Audio Processing Constants
# ================================

SAMPLE_RATE = 16_000
SEGMENT_LEN = 10.0      # in seconds.
HOP_LEN = 5.0           # defines overlap in seconds.
AUDIO_ENCODING = 'PCM_S'
BITS_PER_SAMPLE = 16

# ================================
# Global Experiment Constants
# ================================

# Experiment Labels / Classes
CLASSES = ['speech', 'music', 'noise']

# ================================
# Gold Standard Recordings
# ================================

GOLD_RECORDINGS = [
    '001_NO_RAD_0001.wav',
    '002_NO_RAD_0101.wav',
    '003_NO_RAD_0235.wav',
    '004_NO_RAD_0301.wav'
]

# ================================
# Teacher Model HF Checkpoints
# ================================

AST_MODEL_NAME = 'MIT/ast-finetuned-audioset-10-10-0.4593'
WHISPER_MODEL_SIZE = 'large-v2'
CLAP_MODEL_NAME = 'laion/larger_clap_music_and_speech'
STUDENT_MODEL_ID = 'dams/your-dams-student-checkpoint'
M2D_CLAP_CHECKPOINT = (
        CHECKPOINTS_PATH
        / 'm2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025'
        / 'checkpoint-30.pth'
)


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables.

    This settings class centralizes all paths, Backblaze B2 credentials,
    and project-level parameters used throughout the audio processing
    pipeline. Values are loaded from a `.env` file when present, or from
    system environment variables, allowing reproducible and overrideable
    configuration for local development and team workflows.
    """
    # Backblaze B2
    b2_key_id: str
    b2_application_key: str
    b2_bucket_name: str
    b2_region: str = 'us-east-005'
    b2_endpoint: str = 'https://s3.us-east-005.backblazeb2.com'

    # File paths (can be overridden via .env temporarily, if needed)
    data_root: Path = PROJECT_ROOT / 'data'
    models_path: Path = PROJECT_ROOT / 'models'
    logs_path: Path = PROJECT_ROOT / 'logs'

    # General dataset paths.
    raw_audio_path: Path = data_root / 'raw'
    metadata_path: Path = data_root / 'metadata'
    segments_path: Path = data_root / 'segments'
    gold_labels_path: Path = data_root / 'gold_labels'
    experiments_path: Path = metadata_path / 'experiments'

    # BLOCS paths.
    # raw_audio_blocs_path: Path = raw_audio_path / 'blocs'
    # segments_blocs_path: Path = segments_path / 'blocs'
    # metadata_blocs_path: Path = metadata_path / 'blocs'
    # gold_labels_blocs_path: Path = gold_labels_path / 'blocs'

    # AVA paths.
    # raw_audio_ava_path: Path = raw_audio_path / 'ava'
    # segments_ava_path: Path = segments_path / 'ava'
    # metadata_ava_path: Path = metadata_path / 'ava'
    # gold_labels_ava_path: Path = gold_labels_path / 'ava'

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
    )

@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance.

    The `lru_cache` decorator ensures that application configuration is
    loaded only once from environment variables or the `.env` file.
    Subsequent calls return the same Settings object, providing a single
    shared source of configuration across modules.
    """
    return Settings()
