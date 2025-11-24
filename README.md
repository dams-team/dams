# DAMS: Domain Adaptation for Multi-Label Speech

This repository contains code for **multi-label speech, music, and noise detection** in noisy real-world radio broadcast audio.
Our approach uses a single pretrained AudioSet encoder as a zero-shot teacher to label radio segments, then applies confidence-based pseudo-labeling and fine-tuning to adapt this model to the radio domain. We also benchmark an alternative pretrained encoder (or an encoder-decoder) as a secondary baseline.

The project focuses on a novel, low-resource English spoken corpus called **BLOCS**, which includes a component with naturalistic radio speech from New Orleans, Louisiana.

---

## Overview

Each input $x$ is a fixed-length radio segment, and the model outputs a vector $y \in \{0,1\}^3$ over `[speech, music, noise]`. We train with a multi-label objective using binary cross-entropy with logits, which combines a sigmoid layer with binary cross-entropy so that each class is treated as a yes-or-no decision. At inference time, we apply per-class thresholds to these scores to determine which acoustic events are present in each segment.

---

## Workflow

1. **Segmentation**  
   Long-form radio recordings are resampled to 16 kHz mono, then split into fixed-length segments (10 seconds with 50% overlap). Each segment is tracked in the master metadata table with the fields `raw_file`, `segment_path`, `start_time`, `end_time`, and `split` (`dev`, `test`, or `unlabeled`). All segment WAV files live under a single `data/segments/` directory, and the `split` field in the CSV specifies whether a segment is being used for human-labeled evaluation or will remain in the unlabeled pool for zero-shot and pseudo-labeling.

2. **Zero-Shot Teacher Baseline (AST)**  
   We use the Audio Spectrogram Transformer (AST) model fine-tuned on AudioSet (`MIT/ast-finetuned-audioset-10-10-0.4593`) as the main zero-shot teacher. The model produces 527 AudioSet class scores, which are converted to per-class probabilities and then collapsed into three task labels `[speech, music, noise]` using a fixed index mapping.
   In parallel, we optionally evaluate a second pretrained encoder or encoder–decoder (either wav2vec 2.0 or Whisper-AT) as a comparative model on the same segments. Whisper-AT can be used in a zero-shot audio tagging setup, and wav2vec 2.0 is treated as a feature extractor fine-tuned on the same labels. (Note: this model is used for analysis, not as a teacher for pseudo-labels.) If a labeled radio dev set is available (rows with `label_source = gold`), `evaluate_zero_shot` reports per-class and macro F1 to obtain baseline F1 scores for each class.

3. **Confidence Threshold Estimation for Pseudo-Labeling**  
   To decide which teacher predictions to trust as pseudo-labels, we estimate a confidence threshold per class:
   - With labeled dev data, `calibrate_confidence_thresholds` selects thresholds that achieve a target precision (e.g., 0.85) for each class.
   - Without labeled dev data, `estimate_thresholds_from_unlabeled` runs AST on unlabeled segments and sets thresholds to a chosen upper quantile (for example, the 90th percentile) of the score distribution for each class.

4. **Multilabel Label Sources (Gold, AST Pseudo, VAD+Music)**  
   Using the chosen thresholds, `generate_pseudo_labels` runs the AST teacher over unlabeled radio segments and writes multi-hot labels `[speech, music, noise]` into the same metadata table. Each row is tagged with a `label_source`:
   - `gold` for human-annotated labels
   - `ast_pseudo` for labels derived from AST probabilities
   - `vad_music` for labels derived from VAD + music detectors (e.g., pyannote + PANNs/YamNet), if used  
   This allows experiments that compare or combine different supervision signals without changing the directory layout.
   Each output row has `[speech, music, noise]` in `{0, 1}`:
     - `[1, 0, 0]` for speech only  
     - `[0, 1, 0]` for music only  
     - `[1, 1, 0]` for speech with background music  
     - `[0, 0, 1]` for noise only  
   Segments with no class above its threshold are discarded from pseudo-labeling. The resulting table lives at `data/metadata/blocs_smad_labels.csv`.

5. **Domain Adaptation via Fine-Tuning**  
   We adapt the AST encoder to the radio domain by fine-tuning on selected label sources (e.g., `ast_pseudo` only, or `ast_pseudo` + `vad_music`) using `fine_tune_model`. The classification head is configured for three outputs, and the model is trained with a multilabel objective (`problem_type="multi_label_classification"` with `BCEWithLogitsLoss`) on `[speech, music, noise]` targets.

6. **Evaluation and Analysis**  
   Once a gold-labeled dev and test subset is available, we evaluate both:
   - The original zero-shot AST teacher  
   - The radio-adapted model fine-tuned on pseudo-labels and/or VAD+music labels
   using per-class F1, macro F1, and confusion matrices. Additional analysis (e.g., error breakdown by overlap type or program segment) will be performed in `notebooks/dams_results_figures.ipynb`.

---

## Installation

This project assumes:

- Python 3.9+  
- A working PyTorch install (CPU, MPS, or GPU)  
- Access to Hugging Face Hub for pretrained models
- `ffmpeg` installed on your system for audio I/O and resampling

Clone the repository and install dependencies:

```bash
git clone https://github.com/dams-team/dams.git
cd dams

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```
Fill in .env with your Backblaze B2 credentials (B2_KEY_ID, B2_APPLICATION_KEY, and the target bucket name) before running any data sync scripts.

---

## Project Structure

```text
dams/
│
├── config.py                           # Central paths + env access (B2, etc.)
├── .env.example                        # Template for secrets (B2_KEY_ID, etc.)
├── .gitignore                          # Expand for .env, models, local data, etc.
│
├── utils/
│   └── b2_utils.py                     # Helpers for syncing and downloading from Backblaze B2
│
├── scripts/                            
│   └── blocs_audio_classification.py   # Main pipeline: zero-shot → thresholds → pseudo-labels → fine-tune
│
├── data/
│   ├── raw/                            # Original long-form radio shows (optional local mirror)
│   ├── segments/                       # All segment-level WAVs (dev, test, unlabeled) live here
│   └── metadata/
│       ├──blocs_smad_labels.csv       # file, split (dev, test, unlabeled), label_source, [speech, music, noise], [speech_score, music_score, noise_score]
│       └── experiments/                # Per-run snapshots of labels/predictions
│           ├── ast_pseudo_2025-11-24.csv
│           └── whisper_pseudo_2025-11-25.csv
│
├── models/
│   ├── ast_blocs_finetuned/            # Audio Spectrogram Transformer fine-tuned checkpoints (ignored by Git)
│   ├── w2v2_blocs_finetuned/           # wav2vec 2.0
│   └── w-at_blocs_finetuned/           # Whisper-AT
│
├── notebooks/
│   └── dams_results_figures.ipynb  # Analysis notebook (expects labels/predictions once generated)
│
├── paper/
│   ├── figures/                    # Saved figures from notebooks (confusion matrices, F1 plots, etc.)
│   └── dams_paper.pdf              # Final submitted paper
│
├── requirements.txt                
└── README.md
```
`data/metadata/experiments/` stores per-run CSVs for different teacher models and thresholds, so you can compare setups without overwriting the master manifest.

---

## Data

### Layout and storage

All BLOCS audio for this project lives in a Backblaze B2 bucket rather than in the repository.  
Access is controlled via the credentials in `.env`, while `config.py` and `utils/b2_utils.py` handle the mapping
between remote keys and the local `data/` layout.

We keep the remote structure as a mirror of `data/`:

- Local: `data/raw/...`      ↔  Remote key: `raw/...`
- Local: `data/segments/...` ↔  Remote key: `segments/...`

In other words, the B2 key is simply the local path relative to `data/`.

### Remote vs local paths

On Backblaze B2, long-form radio recordings are stored under a `raw/` prefix. For example:

- `raw/001/001_NO_RAD_0001.wav`
- `raw/001/001_NO_RAD_0002.wav`

When synced locally, these become:

- `data/raw/001/001_NO_RAD_0001.wav`
- `data/raw/001/001_NO_RAD_0002.wav`

Segment-level files are stored under a single `segments/` prefix. For example:

- Remote: `segments/001_NO_RAD_0001_s0001.wav`  
- Local:  `data/segments/001_NO_RAD_0001_s0001.wav`

Here, `s0001`, `s0002`, … denote segment indices for a given raw file (split by 10-second windows with 50% overlap).

### Metadata manifest

The metadata file `data/metadata/blocs_smad_labels.csv` ties everything together.  
Each row corresponds to one segment-level file and includes segmentation info, split assignment, and SMAD labels.

Paths in the CSV are stored **relative to `data/`**, so the code can reconstruct absolute paths by prefixing `data/`:

```text
raw_file,segment_path,start_time,end_time,split,label_source,speech,music,noise,speech_score,music_score,noise_score
raw/001/001_NO_RAD_0001.wav,segments/001_NO_RAD_0001_s0001.wav,0.0,10.0,dev,ast_pseudo,1,1,0,0.93,0.88,0.05
raw/001/001_NO_RAD_0001.wav,segments/001_NO_RAD_0001_s0002.wav,5.0,15.0,unlabeled,ast_pseudo,0,1,0,0.12,0.91,0.07
raw/001/001_NO_RAD_0002.wav,segments/001_NO_RAD_0002_s0001.wav,0.0,10.0,test,gold,1,0,0,1.00,0.02,0.01
