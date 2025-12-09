# data_processing/build_acoustic_stats.py

"""Build the first acoustically enriched SMAD segments manifest.

This script computes basis acoustic stats and flags on pre-segmented audio.
It assumes that the segments are already preprocessed to a fixed sample rate and
mono channel.

Usage:
    python -m data_processing.build_acoustic_stats
"""

from pathlib import Path
from functools import partial

import torch
import torchaudio
from datasets import Dataset

from config import get_settings, SAMPLE_RATE
from utils.dams_types import BLOCS_SMAD_SEGMENTS, BLOCS_SMAD_V1, SegmentManifest

from utils.logger import logger


def preview_rows(ds: Dataset, n: int = 5) -> SegmentManifest:
    """Utility to preview the first n rows of a Dataset as SegmentManifest.

    Note: This function is only for debugging and inspection.
    It materializes the entire dataset into memory.

    Args:
        ds: The Dataset to preview.
        n: Number of rows to return.
    Returns:
        A SegmentManifest list of the first n rows.
    """
    return ds.select(range(n)).to_list()


def _load_segment_waveform(segments_dir: Path, segment_name: str) -> torch.Tensor:
    """Load one segment waveform for analysis.

    Assumes all segments are already saved at SAMPLE_RATE and mono,
    as produced by segment_raw_audio.py.

    Args:
        segments_dir: Directory containing segment audio files.
        segment_name: Filename of the segment to load.

    Returns:
        A 1D torch.Tensor containing the mono waveform samples.

    """
    seg_path = segments_dir / segment_name
    waveform, sr = torchaudio.load(seg_path)

    # Validate preprocessed audio assumption.
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz for {segment_name}"
    assert waveform.size(
        0) == 1, f"Expected mono, got {waveform.size(0)} channels for {segment_name}"

    return waveform.squeeze(0)  # Ensure 1D -> [1, samples] -> [samples]


def _compute_rms_db(waveform: torch.Tensor, min_rms: float = 1e-8) -> float:
    """Compute simple RMS in decibels for a mono waveform.

    Returns a scalar like -20.0 for fairly loud speech or much lower for quiet segments.
    Uses clamping for numerical stability in float32.

    Args:
        waveform: 1D torch.Tensor of audio samples.
        min_rms: Minimum RMS value to avoid log(0).

    Returns:
        RMS value in decibels (dB).
    """
    rms = torch.sqrt(torch.mean(waveform.pow(2)))  # Root Mean Square
    rms = torch.clamp(rms, min=min_rms)  # Avoid log(0).
    rms_db = 20.0 * torch.log10(rms)  # Convert to decibels.
    return float(rms_db)


def _compute_silence_ratio(
    waveform: torch.Tensor,
    threshold_db: float = -60.0,
    min_amplitude: float = 1e-8,
) -> float:
    """Fraction of samples whose energy falls below a fixed dB threshold.

    This is a crude proxy for "how much of this segment is near silence".
    Uses clamping for numerical stability in float32.

    Args:
        waveform: 1D torch.Tensor of audio samples.
        threshold_db: Amplitude threshold in dB below which samples are "silent".
        min_amplitude: Minimum amplitude to avoid log(0).
    Returns:
        Fraction of samples below the silence threshold.
    """
    amplitude = torch.clamp(waveform.abs(), min=min_amplitude)
    amplitude_db = 20.0 * torch.log10(amplitude)
    mask = amplitude_db < threshold_db  # Boolean mask of "silent" samples.
    ratio = mask.float().mean()  # Fraction of samples below threshold.
    return float(ratio)


def _compute_clipping_ratio(
    waveform: torch.Tensor,
    clipping_threshold: float = 0.99,
) -> float:
    """Fraction of samples whose amplitude is at or above a clipping threshold.

    Assumes waveform is float32 in approximately [-1.0, 1.0] after decoding.

    Args:
        waveform: 1D torch.Tensor of audio samples.
        clipping_threshold: Amplitude threshold above which samples are "clipped".
    Returns:
        Fraction of samples above the clipping threshold.
    """
    amplitude = waveform.abs()
    clipped_mask = amplitude >= clipping_threshold
    ratio = clipped_mask.float().mean()
    return float(ratio)


def _compute_zero_crossing_rate(waveform: torch.Tensor) -> float:
    """Zero-crossing rate: fraction of samples where signal crosses zero.

    Useful for distinguishing content types:
    - Speech: typically 0.05-0.15
    - Music: typically 0.02-0.10
    - Noise: often > 0.20

    Higher ZCR may indicate noisy or non-speech content.

    Args:
        waveform: 1D torch.Tensor of audio samples.

    Returns:
        Zero-crossing rate as a fraction of total samples.
    """
    # Count sign changes.
    sign_changes = torch.diff(torch.sign(waveform))
    zero_crossings = (sign_changes != 0).sum()
    zcr = float(zero_crossings) / len(waveform)
    return zcr


def _estimate_snr(waveform: torch.Tensor, frame_size: int = 400) -> float:
    """Rough Signal-to-Noise Rate (SNR) estimate in dB using frame-level energy.

    We treat the 10th percentile of frame energies as a crude noise floor
    and the mean frame energy as the overall signal level. This is not a
    broadcast-standards SNR, just a relative indicator for dataset analysis.

    Args:
        waveform: 1D torch.Tensor of audio samples.
        frame_size: Number of samples per analysis frame.

    Returns:
        Float SNR estimate in decibels (dB).
    """
    energy = waveform.pow(2)

    # If the segment is very short, fall back to a simple global estimate.
    if waveform.numel() < frame_size:
        signal_energy = energy.mean()
        noise_floor = energy.min().clamp_min(1e-8)
    else:
        frames = waveform.unfold(0, frame_size, frame_size)
        frame_energy = frames.pow(2).mean(dim=1)
        signal_energy = frame_energy.mean()
        noise_floor = torch.quantile(frame_energy, 0.1)

    snr_db = 10.0 * torch.log10(signal_energy / (noise_floor + 1e-8))
    return float(snr_db)


def _detect_energy_variance(waveform: torch.Tensor, frame_size: int = 400) -> float:
    """Variance of log10 frame energies.

    Higher variance typically means more bursty dynamics (e.g., turns,
    laughter, ads cutting in and out). Flatter segments (music beds,
    steady noise) tend to have lower variance.

    Args:
        waveform: 1D torch.Tensor of audio samples.
        frame_size: Number of samples per analysis frame.

    Returns
        Float variance of log10 frame energies.
    """
    if waveform.numel() < frame_size:
        # Too short to compute a meaningful variance, treat as flat.
        return 0.0

    frames = waveform.unfold(0, frame_size, frame_size // 2)
    frame_energy = frames.pow(2).mean(dim=1)
    log_energy = torch.log10(frame_energy + 1e-8)
    variance = log_energy.var()
    return float(variance)  # 0.0 for segments shorter than frame_size.


def _add_acoustic_flags(
    batch: dict[str, list],
    segments_dir: Path,
    min_duration_sec: float = 0.5,
    quiet_rms_db: float = -50.0,
    high_silence_ratio: float = 0.8,
    high_clipping_ratio: float = 0.05,
    high_zcr_threshold: float = 0.20,
) -> dict[str, list]:
    """Batch function for Dataset.map.

    Loads each segment in the batch, computes simple stats,
    and adds new columns to the batch dict.

    The incoming batch corresponds to a slice of the SegmentManifest schema.

    Args:
        batch: A batch dictionary with 'segment_path' key.
        segments_dir: Directory containing segment audio files.
        min_duration_sec: Minimum acceptable duration in seconds.
        quiet_rms_db: RMS dB threshold below which segment is "too quiet".
        high_silence_ratio: Silence ratio above which segment is "mostly silence".
        high_clipping_ratio: Clipping ratio above which segment is "heavily clipped".
        high_zcr_threshold: ZCR above which segment is considered "high ZCR".
    Returns:
        The augmented batch dictionary with new acoustic stats and flags.
    """
    duration_list: list[float] = []
    is_too_short_list: list[bool] = []
    rms_db_list: list[float] = []
    is_too_quiet_list: list[bool] = []
    silence_ratio_list: list[float] = []
    is_mostly_silence_list: list[bool] = []
    clipping_ratio_list: list[float] = []
    is_heavily_clipped_list: list[bool] = []
    zcr_list: list[float] = []
    is_high_zcr_list: list[bool] = []
    snr_db_list: list[float] = []
    energy_variance_list: list[float] = []
    had_error_list: list[bool] = []

    segment_names: list[str] = batch["segment_path"]

    for seg_name in segment_names:
        had_error = False
        try:
            waveform = _load_segment_waveform(segments_dir, seg_name)
            duration_sec = len(waveform) / SAMPLE_RATE
            rms_db = _compute_rms_db(waveform)
            silence_ratio = _compute_silence_ratio(waveform)
            clipping_ratio = _compute_clipping_ratio(waveform)
            zcr = _compute_zero_crossing_rate(waveform)
            snr_db = _estimate_snr(waveform)
            energy_variance = _detect_energy_variance(waveform)
        except Exception as e:
            logger.error(f"Error processing {seg_name}: {e}")
            had_error = True
            duration_sec = 0.0
            rms_db = float('-inf')          # Very quiet on error.
            silence_ratio = 1.0             # Pure silence on error.
            clipping_ratio = 0.0            # No clipping on error.
            zcr = float('nan')
            snr_db = float('nan')
            energy_variance = float('nan')  # Don't pollute variance stats.

        rms_db_list.append(rms_db)
        silence_ratio_list.append(silence_ratio)
        clipping_ratio_list.append(clipping_ratio)
        duration_list.append(duration_sec)
        snr_db_list.append(snr_db)
        energy_variance_list.append(energy_variance)
        is_too_quiet_list.append(rms_db < quiet_rms_db)
        is_mostly_silence_list.append(silence_ratio > high_silence_ratio)
        is_heavily_clipped_list.append(clipping_ratio > high_clipping_ratio)
        is_too_short_list.append(duration_sec < min_duration_sec)
        zcr_list.append(zcr)
        is_high_zcr_list.append(zcr > high_zcr_threshold)
        had_error_list.append(had_error)

    batch['duration_sec'] = duration_list
    batch['is_too_short'] = is_too_short_list
    batch['rms_db'] = rms_db_list
    batch['is_too_quiet'] = is_too_quiet_list
    batch['silence_ratio'] = silence_ratio_list
    batch['is_mostly_silence'] = is_mostly_silence_list
    batch['clipping_ratio'] = clipping_ratio_list
    batch['is_heavily_clipped'] = is_heavily_clipped_list
    batch['zero_crossing_rate'] = zcr_list
    batch["is_high_zcr"] = is_high_zcr_list
    batch['snr_db'] = snr_db_list
    batch['energy_variance'] = energy_variance_list
    batch['had_error'] = had_error_list

    return batch


def _log_acoustic_summary(ds_with_flags: Dataset) -> None:
    """Log summary statistics of acoustic flags in the dataset.

    Args:
        ds_with_flags: Dataset with acoustic flags computed.

    Returns:
        None
    """
    n_total = len(ds_with_flags)
    n_too_quiet = sum(ds_with_flags['is_too_quiet'])
    n_mostly_silence = sum(ds_with_flags['is_mostly_silence'])
    n_clipped = sum(ds_with_flags['is_heavily_clipped'])
    n_too_short = sum(ds_with_flags['is_too_short'])
    n_error = sum(ds_with_flags['had_error'])

    logger.info("=== SMAD v1 acoustic statistics ===")
    logger.info(f"Total segments: {n_total}")

    for name, count in [
        ('Too quiet', n_too_quiet),
        ('Mostly silence', n_mostly_silence),
        ('Heavily clipped', n_clipped),
        ('Too short', n_too_short),
        ('Had error', n_error),
    ]:
        pct = 100.0 * count / n_total if n_total else 0.0
        logger.info(f'{name}: {count} ({pct:.1f}%)')

    logger.info('==================================')


def main() -> None:
    settings = get_settings()
    metadata_dir: Path = settings.metadata_path
    segments_dir: Path = settings.segments_path

    base_manifest_path = metadata_dir / BLOCS_SMAD_SEGMENTS
    logger.info(f"Loading base segment manifest from {base_manifest_path}...")
    ds: Dataset = Dataset.load_from_disk(base_manifest_path)

    # Conceptually, each row in `ds` matches the SegmentManifest / SegmentRow schema.
    # We keep it lazy and avoid materializing a full `SegmentManifest` here for scale.
    # rows: SegmentManifest = ds.to_list()  # <-- use for debugging, not in main path.

    # Use Datasets.map to apply _add_acoustic_flags over all batches,
    # binding segments_dir via partial so each batch can load its audio.
    map_fn = partial(_add_acoustic_flags, segments_dir=segments_dir)

    logger.info("Computing acoustic stats and flags for segments...")
    ds_with_flags = ds.map(
        map_fn,
        batched=True,
        batch_size=32,
        desc="Adding acoustic stats and flags",
    )

    out_name = BLOCS_SMAD_V1
    out_path = metadata_dir / out_name

    logger.info(f"Saving SMAD dataset v1 to {out_path}...")
    ds_with_flags.save_to_disk(out_path)

    csv_path = metadata_dir / f"{out_name}.csv"
    ds_with_flags.to_csv(str(csv_path), index=False)
    logger.info(f"✓ Saved SMAD dataset CSV to {csv_path}")

    # Display and log summary statistics.
    _log_acoustic_summary(ds_with_flags)


if __name__ == "__main__":
    main()
