# utils/audioset_mapping.py
"""
Utilities for collapsing raw 527-class AudioSet logits into three DAMS labels:
speech, music, and noise.

The original AudioSet ontology can be found here:
https://research.google.com/audioset/ontology/index.html

See data_processing/apply_ast.py for usage example.
"""
import torch
from functools import lru_cache
from transformers import AutoConfig
from config import AST_MODEL_NAME


@lru_cache(maxsize=1)
def get_ast_config() -> AutoConfig:
    """
    Load the AST config once, using only local files.

    Assumes the model has already been downloaded into the HF cache.
    If you need to populate the cache the first time, you can run a
    small setup script with internet access that calls this without
    local_files_only, then commit to working offline.
    """
    return AutoConfig.from_pretrained(
        AST_MODEL_NAME,
        local_files_only=False, # Set to True after first download.
        cache_dir=None,  # Use default HF cache location.

    )

@lru_cache(maxsize=1)
def get_ast_id2label() -> dict[int, str]:
    cfg = get_ast_config()
    # HF sometimes uses string keys; normalize to int.
    return {int(k): v for k, v in cfg.id2label.items()}


@lru_cache(maxsize=1)
def get_ast_num_labels() -> int:
    return get_ast_config().num_labels


AST_ID2LABEL: dict[int, str] = get_ast_id2label()
AST_NUM_LABELS: int = get_ast_num_labels()

# ======================================================================================
#  1. MUSIC INDICES
# ======================================================================================

# Vocal music: Singing, Choir, Rapping, Humming (27-37)
# Note: "Beatboxing" (218) is covered in the main block.
VOCAL_MUSIC_IDX = list(range(27, 38))

# Whistling (40) is treated as Music (melodic).
WHISTLING_IDX = [40]

# Main Music Block: Instruments (137-138), Specific Instruments (139-217),
# Beatboxing (218), Genres (219-282).
# Range 137 to 282 inclusive.
MUSIC_BLOCK_IDX = list(range(137, 283))

# Combine all music indices.
MUSIC_IDX = sorted(VOCAL_MUSIC_IDX + WHISTLING_IDX + MUSIC_BLOCK_IDX)

# ======================================================================================
#  2. SPEECH INDICES
# ======================================================================================

# Core Speech (0-7): Speech, Male/Female/Child, Conversation, Monologue, Babbling, Synth.
# Loud Vocalizations (8-15): Shout, Yell, Scream, Whisper.
# Laughter & Crying (16-26): Laughter, Giggle, Crying, Sobbing, Sigh, Moan.
SPEECH_CORE_IDX = list(range(0, 27))

# Non-Musical Vocalizations (38-39): Groan, Grunt.
SPEECH_VOCAL_IDX = list(range(38, 40))

# Respiratory & Mouth Sounds (41-50, 58-59):
# 41-50: Breathing, Wheeze, Snore, Gasp, Pant, Snort, Cough, Throat clear, Sneeze, Sniff.
# 58-59: Burping, Hiccup (Vocal tract interruptions).
SPEECH_RESPIRATORY_IDX = list(range(41, 51)) + [58, 59]

# Combine all speech
SPEECH_IDX = sorted(SPEECH_CORE_IDX + SPEECH_VOCAL_IDX + SPEECH_RESPIRATORY_IDX)

# ======================================================================================
#  3. NOISE INDICES (Includes everything else)
# ======================================================================================
# Includes:
# - Bodily noises not from vocal tract (Chewing, Fart, Stomach rumble, Heartbeat)
# - Human movement (Run, Walk, Clapping)
# - Crowd/Hubbub (Index 70)
# - Nature, Animals, Vehicles, Tools, Sirens, etc.
# - Telephone Ringing (389, 390)

# Assuming AST_NUM_LABELS is 527 and indexed 0-AST_NUM_LABELS-1.
ALL_IDX = set(range(AST_NUM_LABELS))
NOISE_IDX = sorted(list(ALL_IDX - set(SPEECH_IDX) - set(MUSIC_IDX)))

# ================================
#  Default settings for collapsing
# ================================

AST_POOLING = 'noisy_or'
AST_THRESHOLD = 0.7          # Default threshold for DAMS decisions. Deprecated.
AST_SPEECH_THRESHOLD = 0.60  # Expanded speech sensitivity.
AST_MUSIC_THRESHOLD  = 0.40  # Expanded music sensitivity.

# ================================
#  Collapsing function
# ================================

def collapse_audioset_logits(
    logits: torch.Tensor,
    speech_idx=SPEECH_IDX,
    music_idx=MUSIC_IDX,
    noise_idx=NOISE_IDX,
    pooling: str = "noisy_or",
) -> torch.Tensor:
    """
    Collapse 527-dimensional AudioSet logits into 3 DAMS scores:
        [speech_score, music_score, noise_score]

    logits:
        Either (527,) or (batch, 527) tensor of raw AudioSet logits.

    Note: Noise includes silence, ambient sounds, and all non-speech/non-music
          audio events from AudioSet ontology.

    pooling:
        'max'       – strongest evidence
        'mean'      – average evidence
        'noisy_or'  – probabilistic union (recommended)
    Returns:
        (batch, 3) tensor of scores in [0, 1], in order
        [speech_score, music_score, noise_score]
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # Slice out the relevant logits.
    speech_slice, music_slice, noise_slice = (
        logits[:, speech_idx], logits[:, music_idx], logits[:, noise_idx]
    )

    if pooling == "noisy_or":  # for multi-label settings.
        # Convert logits to probabilities.
        p_s, p_m, p_n = (torch.sigmoid(speech_slice), torch.sigmoid(music_slice),
                         torch.sigmoid(noise_slice))

        # Noisy-OR: 1 − ∏(1 − p_i)
        s = 1.0 - (1.0 - p_s).prod(dim=1)
        m = 1.0 - (1.0 - p_m).prod(dim=1)
        n = 1.0 - (1.0 - p_n).prod(dim=1)

    elif pooling == "max":
        s, m, n  = (
            torch.sigmoid(speech_slice.max(dim=1).values),
            torch.sigmoid(music_slice.max(dim=1).values),
            torch.sigmoid(noise_slice.max(dim=1).values)
        )
    elif pooling == "mean":
        s, m, n = (
        torch.sigmoid(speech_slice.mean(dim=1)),
        torch.sigmoid(music_slice.mean(dim=1)),
        torch.sigmoid(noise_slice.mean(dim=1))
        )

    else:
        raise ValueError("pooling must be 'max', 'mean', or 'noisy_or'")

    return torch.stack([s, m, n], dim=1)


def get_ast_label_table() -> list[dict[str, object]]:
    """ Get a table of AudioSet labels with their index and group (speech/music/noise)."""
    rows: list[dict[str, object]] = []
    for idx, label in AST_ID2LABEL.items():
        if idx in SPEECH_IDX:
            group = 'speech'
        elif idx in MUSIC_IDX:
            group = 'music'
        elif idx in NOISE_IDX:
            group = 'noise'
        else:
            group = None
        rows.append({'index': idx, 'label': label, 'group': group})
    return rows
