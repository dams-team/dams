# config.py

"""Central configuration module for the DAMS audio processing project.

Defines global constants, model identifiers, and filesystem paths used across
data processing, teacher inference, and training components.

Settings loads environment-dependent values from a .env file or environment
variables. Derived paths are computed so overrides remain consistent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_PATH = PROJECT_ROOT / 'checkpoints' # Directories for non-HF checkpoints.

# ================================
# Audio Processing Constants
# ================================

SAMPLE_RATE = 16_000
SEGMENT_LEN = 10.0          # in seconds.
HOP_LEN = 5.0               # seconds (overlap = SEGMENT_LEN - HOP_LEN)
AUDIO_ENCODING = 'PCM_S'
BITS_PER_SAMPLE = 16

# ================================
# Global Experiment Constants
# ================================

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
    """Application-wide configuration loaded from environment variables."""

    # Backblaze B2
    b2_key_id: str
    b2_application_key: str
    b2_bucket_name: str
    b2_region: str = 'us-east-005'
    b2_endpoint: str = 'https://s3.us-east-005.backblazeb2.com'

    # Label Studio (optional)
    ls_url: str | None = None
    ls_api_key: str | None = None

    # Roots (overrideable)
    data_root: Path = Field(default=PROJECT_ROOT / 'data')
    models_path: Path = Field(default=PROJECT_ROOT / 'models')
    logs_path: Path = Field(default=PROJECT_ROOT / 'logs')
    reports_path: Path = Field(default=PROJECT_ROOT / 'reports')

    # Versioning knobs (overrideable)
    dataset_id: str = Field(default='blocs_smad')
    manifest_version: str = Field(default='v1')  # schema/content version
    run_id: str = Field(default='dev')  # experiment/run id (e.g., 2025-12-01_a)

    # ===============================
    # Derived paths
    # ===============================

    @computed_field
    @property
    def raw_audio_path(self) -> Path:
        return self.data_root / 'raw'

    @computed_field
    @property
    def metadata_path(self) -> Path:
        return self.data_root / 'metadata'

    @computed_field
    @property
    def segments_path(self) -> Path:
        return self.data_root / 'segments'

    @computed_field
    @property
    def gold_labels_path(self) -> Path:
        return self.data_root / 'gold_labels'

    @computed_field
    @property
    def experiments_path(self) -> Path:
        return self.metadata_path / self.dataset_id / 'experiments'

    @computed_field
    @property
    def predictions_path(self) -> Path:
        return self.reports_path / 'label_efficiency' / 'preds'

    # ===============================
    # Versioned folders
    # ===============================

    @computed_field
    @property
    def manifests_root(self) -> Path:
        """Root directory for dataset-scoped, versioned metadata artifacts.

        Example: data/metadata/blocs_smad/v1/...
        """
        return self.metadata_path / self.dataset_id

    def manifest_dir(self, version: str | None = None) -> Path:
        """Return the manifest directory for a given version."""
        ver = version or self.manifest_version
        return self.manifests_root / ver

    @computed_field
    @property
    def runs_root(self) -> Path:
        return self.reports_path / 'runs'

    def run_dir(self, run_id: str | None = None, version: str | None = None) -> Path:
        """Return the run directory for a given run ID."""
        rid = run_id or self.run_id
        vid = version or self.manifest_version
        return self.runs_root / vid / rid


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
