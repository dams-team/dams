# utils/config_utils.py

"""
Configuration utilities for DAMS BLOCS.

This module provides functions to load and manage configuration settings
for the DAMS BLOCS system.

Usage:
    from utils.config_utils import load_config, get_config_value
"""
import sys
from pathlib import Path
from config import get_settings


def ensure_data_dirs() -> None:
    """Ensure that necessary data directories exist.

    This function checks for the existence of required data directories:
    - Raw audio blocs path: holds original audio files
    - Segments path: holds audio segments
    - Metadata path: holds acoustic and teacher manifests and CSVs
    """
    s = get_settings()
    for p in [s.raw_audio_path, s.segments_path, s.metadata_path]:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    """Load environment variables from the project .env file.

    This helper ensures that Jupyter notebooks and standalone scripts
    have access to the same environment variables as CLI tools. It loads
    the `.env` file from the project root directory (one level up from `utils/`).

    Safe to call multiple times — it will not override existing variables.
    """
    from dotenv import load_dotenv

    # Compute the project root from this file's location
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / ".env"

    # Load only if the file exists
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)


def add_project_root_to_path() -> None:
    """Ensure project root is in sys.path for notebook imports."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
