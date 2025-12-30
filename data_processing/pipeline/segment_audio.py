# data_processing/pipeline/segment_audio.py

"""Process long-form raw audio recordings into overlapping segments.

Reads all WAV files from a configured source directory, segments them with
configurable length and overlap, and saves segments with a tracking manifest.

Usage:
    python -m data_processing.pipeline.segment_audio --corpus blocs
    python -m data_processing.pipeline.segment_audio --corpus
"""

from pathlib import Path
from typing import Iterator

import torchaudio
import pandas as pd

from config import (
    get_settings,
    AUDIO_ENCODING,
    BITS_PER_SAMPLE,
    HOP_LEN,
    SAMPLE_RATE,
    SEGMENT_LEN,
)
from utils.dams_types import SegmentBaseRow

from utils.artifacts import (
    SEGMENTS_CSV,
    V1
)

from utils.logger import logger

from dataclasses import dataclass

@dataclass
class SegmentPaths:
    raw_dir: Path           # data/raw/ - input audio files
    segments_dir: Path      # data/segments/ - output segments
    csv_path: Path          # v1/segments.csv  - CSV manifest


def segment_corpus(corpus: str) -> SegmentPaths:
    """Segment all raw audio files in the specified corpus into overlapping segments.
    Cuts long-form recordings into overlapping segments, and save the segments as
    WAV files. Then write a HF manifest dataset and CSV.

    Args:
        corpus (str): The corpus to process (default = 'blocs').

    Returns:
        SegmentPaths: Paths and filenames used for segmentation outputs.
    """
    settings = get_settings()
    # Get the v1 manifest directory.


    if corpus == 'blocs':
        return SegmentPaths(
            raw_dir=settings.raw_audio_path,
            segments_dir=settings.segments_path,
            csv_path=settings.manifest_dir(V1) / SEGMENTS_CSV,
        )
    # Note: Additional corpora can be added here.
    else:
        raise ValueError(f'Unknown corpus: {corpus}')


def iter_raw_files(raw_dir: Path) -> Iterator[Path]:
     """Yield all raw WAV files under the configured raw directory."""
     for wav_path in raw_dir.rglob("*.wav"):
         yield wav_path


def make_segments_for_file(wav_path: Path, segments_dir: Path) -> list[SegmentBaseRow]:
    """
    Load one long-form recording, cut into overlapping segments, write each
    segment as a WAV, and return manifest rows.
    """
    waveform, sr = torchaudio.load(wav_path)  # waveform: (channels, samples)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Collapse to mono, if needed.
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    num_samples = waveform.shape[-1]
    total_duration = num_samples / SAMPLE_RATE

    base = wav_path.stem                # waveform: (channels, samples)
    raw_relative_path = wav_path.name   # store filename only

    rows: list[SegmentBaseRow] = []
    seg_idx = 1
    current_time = 0.0

    segments_dir.mkdir(parents=True, exist_ok=True)

    while current_time < total_duration:
        start_time = current_time
        end_time = min(current_time + SEGMENT_LEN, total_duration)

        # Drop very tiny segments at the end.
        if end_time - start_time < 1.0:
            break

        start_sample = int(start_time * SAMPLE_RATE)
        end_sample = int(end_time * SAMPLE_RATE)

        segment_waveform = waveform[..., start_sample:end_sample]
        seg_name = f'{base}_s{seg_idx:04d}.wav'
        seg_path = segments_dir / seg_name

        torchaudio.save(
            str(seg_path),
            segment_waveform,
            SAMPLE_RATE,
            encoding=AUDIO_ENCODING,
            bits_per_sample=BITS_PER_SAMPLE,
        )

        rows.append(
            SegmentBaseRow(
                raw_file=raw_relative_path,
                segment_path=seg_name,
                start_time=float(start_time),
                end_time=float(end_time),
            )
        )

        seg_idx += 1
        current_time += HOP_LEN

    return rows


def main() -> None:

    import argparse
    parser = argparse.ArgumentParser(
        description='Segment raw audio recordings into overlapping segments.'
    )
    parser.add_argument(
        '--corpus',
        choices=['blocs', 'ava'],
        required=True,
        default='blocs',
        help="Corpus to segment.",
    )
    args = parser.parse_args()

    # Get corpus and corpus file paths.
    corpus = args.corpus
    paths = segment_corpus(corpus)

    paths.segments_dir.mkdir(parents=True, exist_ok=True)
    segments_manifest: list[SegmentBaseRow] = []  # Holds rows from processed raw files.

    for wav_path in iter_raw_files(paths.raw_dir):
        try:
            logger.info(f'Processing {wav_path} for corpus={corpus!r}...')
            rows = make_segments_for_file(wav_path, paths.segments_dir)
            segments_manifest.extend(rows)
        except Exception as e:
            logger.error(f'Failed to process {wav_path}: {e}')
            continue

    # Save segments CSV to v1 directory.
    paths.csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(segments_manifest)
    df.to_csv(paths.csv_path, index=False)
    logger.info(f'✓ Saved segment manifest CSV to {paths.csv_path}')


if __name__ == '__main__':
    main()
