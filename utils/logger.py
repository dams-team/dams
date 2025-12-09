# utils/logger.py

"""
Logger configuration using Loguru.

Usage:
    from utils.logger import logger
"""

from loguru import logger
from pathlib import Path

# Additional imports for pseudo label stats logging.
from datasets import Dataset
import pandas as pd

from utils.dams_types import (
    SPEECH,
    MUSIC,
    NOISE,
    SPEECH_SCORE,
    MUSIC_SCORE,
    NOISE_SCORE,
)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "dams.log",
    rotation="100 MB",
    backtrace=True,
    diagnose=False,
    enqueue=True
)


def log_pseudo_label_stats(ds: Dataset, teacher_name: str = "teacher") -> None:
    """
    Log summary statistics for pseudo labels (speech, music, noise) produced by a teacher model.
    Args:
        ds - Hugging Face Dataset containing label and score columns.
        teacher_name (str) Name of the teacher used for logging context.
    """
    # Pull only the label and score columns into a small DataFrame
    df = ds.to_pandas()[[SPEECH, MUSIC, NOISE, SPEECH_SCORE, MUSIC_SCORE, NOISE_SCORE]]
    total = len(df)

    speech_pos = int(df[SPEECH].sum())
    music_pos = int(df[MUSIC].sum())
    noise_pos = int(df[NOISE].sum())

    logger.info(f"=== {teacher_name} label summary ===")
    logger.info(f"Total segments: {total}")
    logger.info(f"Speech=1: {speech_pos} ({speech_pos / total:.1%})")
    logger.info(f"Music=1:  {music_pos} ({music_pos / total:.1%})")
    logger.info(f"Noise=1:  {noise_pos} ({noise_pos / total:.1%})")

    both_sm = int(((df[SPEECH] == 1) & (df[MUSIC] == 1)).sum())
    sm_only = int(((df[SPEECH] == 1) & (df[MUSIC] == 0)).sum())
    m_only = int(((df[SPEECH] == 0) & (df[MUSIC] == 1)).sum())
    none_sm = int(((df[SPEECH] == 0) & (df[MUSIC] == 0)).sum())

    sn = int(((df[SPEECH] == 1) & (df[NOISE] == 1) & (df[MUSIC] == 0)).sum())
    mn = int(((df[MUSIC] == 1) & (df[NOISE] == 1) & (df[SPEECH] == 0)).sum())
    smn = int(((df[SPEECH] == 1) & (df[MUSIC] == 1) & (df[NOISE] == 1)).sum())
    noise_only = int(((df[NOISE] == 1) & (df[SPEECH] == 0) & (df[MUSIC] == 0)).sum())

    logger.info(f"Speech–music combinations ({teacher_name}):")
    logger.info(f"  speech=1, music=1: {both_sm}")
    logger.info(f"  speech=1, music=0: {sm_only}")
    logger.info(f"  speech=0, music=1: {m_only}")
    logger.info(f"  speech=0, music=0: {none_sm}")

    logger.info(f"Speech–noise combinations ({teacher_name}):")
    logger.info(f"  speech=1, noise=1 (music=0): {sn}")
    logger.info(f"  speech=0, noise=1 (music=0): {noise_only}")

    logger.info(f"Music–noise combinations ({teacher_name}):")
    logger.info(f"  music=1, noise=1 (speech=0): {mn}")

    logger.info(f"All three labels=1 ({teacher_name}): {smn}")

__all__ = ["logger", "log_pseudo_label_stats"]
