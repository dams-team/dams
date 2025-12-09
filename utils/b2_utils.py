# utils/b2_utils.py

"""Helpers for syncing BLOCS audio from Backblaze B2 (S3-compatible) to the local dirs.

Environment variables required:
    B2_KEY_ID            - Backblaze key ID (S3 access key)
    B2_APPLICATION_KEY   - Backblaze application key (S3 secret key)
    B2_BUCKET_NAME       - Bucket name, e.g. "blocs-audio"

Optionally, you can also set:
    B2_ENDPOINT          - Backblaze endpoint URL
    B2_REGION            - Backblaze region

We assume the following layout in B2:
    raw/        -> original long-form radio shows
    segments/   -> all segment-level WAVs (dev, test, unlabeled), split is in metadata
    metadata/   -> all metadata files (CSV, JSONL, etc.)

Locally, these are mirrored under:
    data/raw/
    data/segments/
    data/metadata/
"""

from pathlib import Path
import boto3
from config import get_settings

settings = get_settings()


def _get_b2_bucket():
    """Create and return a B2 S3 Bucket resource."""
    session = boto3.session.Session(
        aws_access_key_id=settings.b2_key_id,
        aws_secret_access_key=settings.b2_application_key,
        region_name=settings.b2_region,
    )
    s3 = session.resource('s3', endpoint_url=settings.b2_endpoint)
    return s3.Bucket(settings.b2_bucket_name)


def sync_prefix_to_local(prefix: str, local_root: str) -> None:
    """Sync all objects under B2 `prefix` into local_root.

    Example:
        sync_prefix_to_local('dev/', 'data/dev')

    This is a simple one-way "download if missing" sync.
    """
    bucket = _get_b2_bucket()
    local_root_path = Path(local_root)
    local_root_path.mkdir(parents=True, exist_ok=True)

    print(f'Syncing prefix "{prefix}" -> {local_root_path}')

    for obj in bucket.objects.filter(Prefix=prefix):
        # Skip "directory" placeholders
        if obj.key.endswith('/'):
            continue

        rel_key = obj.key[len(prefix):] if obj.key.startswith(prefix) else obj.key
        dest_path = local_root_path / rel_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists():
            # Simple cache: skip existing files
            continue

        print(f'  Downloading {obj.key} -> {dest_path}')
        bucket.download_file(obj.key, str(dest_path))

    print(f'✓ Finished syncing {prefix}')


def upload_file_to_b2(local_path: str | Path, remote_key: str) -> None:
    """Upload a single local file to B2 under the given key.

    Example:
        upload_file_to_b2('data/segments/dev/foo.wav', 'segments/dev/foo.wav')
    """
    bucket = _get_b2_bucket()
    local_path = Path(local_path)

    if not local_path.is_file():
        raise FileNotFoundError(f'Local file not found: {local_path}')

    # Avoid leading slash in key
    remote_key = remote_key.lstrip('/')

    print(f'Uploading {local_path} -> b2://{settings.b2_bucket_name}/{remote_key}')
    bucket.upload_file(str(local_path), remote_key)
    print('✓ Upload complete')


def sync_local_to_prefix(local_root: str | Path, prefix: str) -> None:
    """Upload all files under local_root into the B2 `prefix`.

    Example:
        sync_local_to_prefix(settings.segments_path, 'segments/')
    """
    bucket = _get_b2_bucket()
    local_root_path = Path(local_root)

    if not local_root_path.exists():
        raise FileNotFoundError(f'Local root not found: {local_root_path}')

    print(f'Syncing local {local_root_path} -> prefix "{prefix}" in B2')

    for path in local_root_path.rglob('*'):
        if path.is_dir():
            continue

        rel_path = path.relative_to(local_root_path)
        remote_key = f'{prefix.rstrip("/")}/{rel_path.as_posix()}'

        print(f'  Uploading {path} -> {remote_key}')
        bucket.upload_file(str(path), remote_key)

    print(f'✓ Finished uploading {local_root_path} to "{prefix}"')


def prepare_local_data() -> None:
    """Convenience helper to sync the standard prefixes:
        raw/       -> settings.raw_dir
        segments/  -> settings.segments_dir
        metadata/   -> settings.metadata_path
    """
    sync_prefix_to_local('raw/', settings.raw_audio_path)
    sync_prefix_to_local('segments/', settings.segments_path)
    sync_prefix_to_local('metadata/', settings.metadata_path)


def push_segments_to_b2() -> None:
    """Convenience helper to upload all local segments to B2."""
    sync_local_to_prefix(settings.segments_path, 'segments/')


def sync_metadata_from_b2() -> None:
    """Sync all metadata files from B2 `metadata/` into the local metadata directory."""
    sync_prefix_to_local('metadata/', settings.metadata_path)


def push_metadata_to_b2() -> None:
    """Upload local metadata files to B2 under the `metadata/` prefix.
    Use with care; you generally want this to run from a controlled machine.
    """
    sync_local_to_prefix(settings.metadata_path, 'metadata/')
