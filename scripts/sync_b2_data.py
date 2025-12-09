# scripts/sync_b2_data.py

"""Creates local data directories and syncs data down from Backblaze B2 cloud storage.

This script downloads:
    raw/        -> data/raw/
    segments/   -> data/segments/
    metadata/   -> data/metadata/

Usage:
    python scripts/sync_b2_data.py
"""

from utils.b2_utils import prepare_local_data
from utils.config_utils import ensure_data_dirs

if __name__ == "__main__":
    ensure_data_dirs()
    prepare_local_data()
