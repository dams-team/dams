# DAMS: Domain Adaptation for Multi-Label Speech

This repository contains code for **multi-label detection of overlapping speech, music, and noise** in real-world radio broadcasts.
The pipeline segments long-form radio audio into fixed-length windows, runs 
multiple pretrained teachers (AudioSet taggers such as AST and Whisper-AT, 
plus CLAP-family audio–text models) to score each segment, then uses 
confidence-based pseudo-labeling. In the future, student fine-tuning will be 
used to adapt models to the selected radio domain.

The project focuses on a novel, low-resource English spoken corpus called 
**BLOCS**, which includes a component with naturalistic radio speech from 
the Deep South.

---

## Overview

Each input $x$ is a fixed-length radio segment, and the model outputs a vector $y \in \{0,1\}^3$ over `[speech, music, noise]`. We train with a multi-label objective using binary cross-entropy with logits, which combines a sigmoid layer with binary cross-entropy so that each class is treated as a yes-or-no decision. At inference time, we apply per-class thresholds to these scores to determine which acoustic events are present in each segment.

In addition to AudioSet-style teachers (AST and Whisper-AT), we include two state-of-the-art contrastive audio–text models (CLAP and M2D-CLAP) as strong zero-shot baselines, using natural-language definitions of speech, music, and noise.

---

## Quickstart

If you have Backblaze B2 credentials and access to the BLOCS bucket, you can 
run a minimal end-to-end pass from the root directory as follows:

1. Create and activate a virtual environment, install dependencies, and configure `.env` as described in the Installation section.
2. Sync audio and metadata from Backblaze B2 into the local `data/` folder:
   ```bash
   python -m scripts.sync_b2_data
   ```
3. (Optional) Compute acoustic statistics for normalization and EDA:
   ```bash
   python -m data_processing.build_acoustic_stats
   ```
4. Run the AST teacher on a subset of segments (for example, dev and test splits) to generate scores and pseudo-labels:
   ```bash
   python -m data_processing.teachers.apply_ast
   ```
5. Open the notebooks to inspect segment distributions and teacher behavior:
   - `notebooks/01_smad_segments_eda.ipynb`
   - `notebooks/02_ast_teacher_sanity.ipynb`
   - `notebooks/03_teacher_comparison.ipynb`

This quickstart does not require the student model code to be finished; it exercises the data layout, teachers, and analysis workflow end-to-end.

---

## Workflow

1. **Segmentation and metadata**  
   Long-form radio recordings are resampled to 16 kHz mono, then split into fixed-length segments (10 seconds with 50% overlap). Each segment is tracked in the master metadata table `data/metadata/blocs_smad_segments.csv` with the fields `raw_file`, `segment_path`, `start_time`, `end_time`, `split`, and placeholder label fields (`label_source`, `speech`, `music`, `noise`, plus optional scores). All segment WAV files live under a single `data/segments/` directory, and the `split` field (`train`, `dev`, `test`, or `unlabeled`) determines whether a segment is used for human-labeled evaluation or remains in the unlabeled pool for teacher models and pseudo-labeling.


2. **AudioSet-based teachers (Audio Spectrogram Transformer and Whisper-AT)**  
   We use the Audio Spectrogram Transformer (AST) model fine-tuned on AudioSet and Whisper-AT as AudioSet-style teachers. Both models predict 527 AudioSet class scores for each segment. We convert logits to probabilities, then collapse them into the task labels `[speech, music, noise]` using a shared AudioSet index mapping defined in `utils/audioset_mapping.py`. For each segment and each teacher, we store pooled scores (`speech_score`, `music_score`, `noise_score`) and derived binary labels in Hugging Face datasets and per-run CSVs under `data/metadata/blocs_smad_segments/` and `data/metadata/experiments/`. When calibrated thresholds are available (see below), we can write pseudo-labels back into the master manifest with `label_source = ast_pseudo` or `label_source = whisper_pseudo`.


3. **CLAP family zero-shot teachers (CLAP and M2D-CLAP)**  
   We also evaluate two contrastive audio–text models as strong zero-shot baselines. `apply_clap.py` uses a Hugging Face CLAP pipeline with three natural-language prompts describing speech, music, and noise; the model returns scores per prompt that we threshold to obtain multi-label decisions. `apply_m2d.py` runs M2D-CLAP by embedding the same prompts with the text encoder, computing cosine similarity with audio embeddings, and thresholding the resulting scores. Both models operate over the same BLOCS segments and produce labels and scores tagged with `label_source = clap_zero_shot` and `label_source = m2d_zero_shot`, respectively. These artifacts are saved as versioned Hugging Face datasets and CSVs in `data/metadata/blocs_smad_segments/` and `data/metadata/experiments/`, and are used as additional teacher signals and zero-shot baselines.


4. **Confidence threshold estimation for pseudo-labeling**  
   To decide which teacher predictions to trust as pseudo-labels, we estimate a confidence threshold per class and per teacher:
   - With labeled dev data (`label_source = gold`), a calibration routine (e.g., `calibrate_confidence_thresholds`) selects thresholds that achieve a target precision (for example, 0.85) for each class.
   - Without labeled dev data, an unsupervised routine (e.g., `estimate_thresholds_from_unlabeled`) runs a teacher on unlabeled segments and sets thresholds to an upper quantile (for example, the 90th percentile) of that teacher’s score distribution per class.

   In the current code, AST thresholds are the canonical choice and can optionally be reused for Whisper-AT and CLAP-family models, but nothing prevents calibrating each teacher separately.


5. **Multilabel label sources and pseudo-label generation**  
   Using the chosen thresholds, `generate_pseudo_labels` applies one or more teachers to unlabeled radio segments and writes multi-hot labels `[speech, music, noise]` into the master metadata table. Each row is tagged with a `label_source`:
   - `none` for unlabeled segments (default)
   - `gold` for human-labeled segments (dev/test only)
   - `ast_pseudo` for labels derived from AST probabilities
   - `whisper_pseudo` for labels derived from Whisper-AT probabilities
   - `clap_zero_shot` for labels derived from CLAP zero-shot scores
   - `m2d_zero_shot` for labels derived from M2D-CLAP zero-shot scores

   This makes it possible to compare or combine different supervision signals without changing the directory layout. Each output row has `[speech, music, noise]` in `{0, 1}`:
     - `[1, 0, 0]` for speech only  
     - `[0, 1, 0]` for music only  
     - `[1, 1, 0]` for speech with background music  
     - `[0, 0, 1]` for noise only
     - `[1, 0, 1]` for speech with background noise  
     - `[0, 1, 1]` for music with background noise  
     - `[1, 1, 1]` for speech with music and noise
     
    Segments with no class above its threshold are excluded from that teacher’s pseudo-label set. The resulting labeled table is stored as `data/metadata/blocs_smad_labels.csv`, and per-run snapshots stay under `data/metadata/experiments/`.


6. **Domain adaptation via student fine-tuning**  
   We adapt a student classifier (for example, an AST-based encoder plus classification head) to the BLOCS radio domain by fine-tuning on selected label sources, such as `gold` plus one or more pseudo-label sources (`ast_pseudo`, `whisper_pseudo`, `clap_zero_shot`, `m2d_zero_shot`). The head is configured for three outputs, and the model is trained with a multilabel objective (`BCEWithLogitsLoss`) on `[speech, music, noise]` targets. This code will live under `models/` and is under active development.


7. **Evaluation and notebook-based analysis**  
   Once a gold-labeled dev and test subset is available, we evaluate both the original teachers and any fine-tuned students using per-class F1, macro F1, and confusion matrices. Agreement and error analyses by overlap type, program segment, and teacher combination are run in the Jupyter notebooks. Currently, `notebooks/01_smad_segments_eda.ipynb`, `notebooks/02_ast_teacher_sanity.ipynb`, and `notebooks/03_teacher_comparison.ipynb` are the main entry points for analysis; `notebooks/04_student_error_analysis.ipynb` and `notebooks/05_paper_figures.ipynb` are placeholders for future student and paper-figure work.


---

## Installation

This project assumes:

- Python 3.12+  
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
Fill in .env with your Backblaze B2 credentials (B2_KEY_ID, 
B2_APPLICATION_KEY, B2_BUCKET_NAME, and B2_ENDPOINT). Then run the sync 
script to populate local data directories from the root directory:

```bash
python -m scripts.sync_b2_data
```
Data will be downloaded into `data/raw/` (long-form shows) and `data/segments/` (10-second segments) as configured in `config.py`.

---

## Project Structure

```text
dams/
│
├── .env.example                        # Template for secrets (B2_KEY_ID, B2_ACCESS_KEY.)
├── .gitignore                          # models, local experiments, etc.
├── config.py                           # Central paths + env access (B2, etc.)
│
├── checkpoints/
│   └── m2d_clap_vit_base/              # Pretrained M2D-CLAP weights
│
├── data/
│   ├── raw/                            # Original long-form radio shows (optional local mirror)
│   ├── segments/                       # All segment-level WAVs (dev, test, unlabeled) live here
│   └── metadata/
│       ├──blocs_smad_segments.csv      # file, splits, label_source, [speech, music, noise], [speech_score, music_score, noise_score]
│       ├──blocs_smad_segments/         # Versioned Pyarrow-backed HF Datasets for each run (v1, v2, Final, etc.)
│       ├──blocs_gold_labels.csv        # Reserved for human-annotated labels for dev/test segments 
│       └── experiments/                # Optional per-run CSV snapshots (ignored by git), for example:
│           ├── ast_pseudo_DEVTEST_YYYYMMDD.csv
│           └── whisper_pseudo_DEVTEST_YYYYMMDD.csv
│
├── data_processing/
│   ├── __init__.py
│   ├── build_acoustic_stats.py         # Compute dataset-level acoustic stats for normalization
│   └── teachers/
│       ├── __init__.py
│       ├── apply_ast.py                # Supervised AudioSet Transformer teacher
│       ├── apply_clap.py               # Zero-shot CLAP audio tagging teacher
│       ├── apply_m2d.py                # Zero-shot M2D-CLAP audio tagging teacher
│       └── apply_whisper.py            # Whisper-AT audio tagging teacher
|
├── logs/
│   └── dams.log                        # General log file
│
├── models/                             # Reserved for fine-tuned model 
definitions
│   ├── __init__.py   
│   ├── TBD: SMAD fine-tuned model definitions (dataset, eval, mode, training loop) 
│   ├── encoders/                       # Encoder architectures 
│   ├── heads/                          # Classification head definitions
│   └── losses/                         # Loss functions 
│   
├── notebooks/
│   ├── 01_smad_segments_eda.ipynb      # EDA of BLOCS SMAD segments, acoustic stats, and gold distribution
│   ├── 02_ast_teacher_sanity.ipynb     # Supervised and zero-shot AST teacher confidence threshold calibration
│   ├── 03_teacher_comparison.ipynb     # Compare teacher performances
│   ├── 04_student_error_analysis.ipynb # Reserved for student errors vs teacher
│   └── 05_paper_figures.ipynb          # Reserved fir final results figures for paper
│
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py                  # Audio loading, resampling, segmentation
│   ├── audioset_mapping.py             # AudioSet class index mapping to [speech, music, noise]
│   ├── b2_utils.py                     # Helpers for syncing and downloading from Backblaze B2
│   ├── config_utils.py                 # Load .env and config.py settings
│   ├── dams_types.py                   # Type definitions and dataclasses
│   ├── logger.py                       # General Logging setup
│   ├── portable_m2d.py                 # M2D feature extractor for M2D-CLAP
│   └── timing.py                       # Timing decorators for profiling
│
├── scripts/                            
│   └── sync_b2_data.py                 # Sync local data/ from Backblaze B2 bucket
│
├── paper/
│   ├── figures/                        # Saved figures from notebooks (confusion matrices, F1 plots, etc.)
│   └── dams_paper.pdf                  # Final submitted paper
│
├── requirements.txt                
└── README.md

Main pipeline: segments → teachers (AST, Whisper AT, CLAP family) → thresholds → pseudo-labels → student fine-tune → eval
```
`data/metadata/experiments/` is an optional directory for per-run CSV snapshots. If you prefer, you can keep teacher CSVs directly under `data/metadata/` and only use the versioned HF datasets under `data/metadata/blocs_smad_segments/`.

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

Segment-level files are stored under a single segments/ prefix. For example:

- Remote: `segments/001_NO_RAD_0001_s0001.wav`  
- Local:  `data/segments/001_NO_RAD_0001_s0001.wav`

Here, `s0001`, `s0002`, … denote segment indices for a given raw file (split by 10-second windows with 50% overlap).

### Metadata manifest

The metadata file `data/metadata/blocs_smad_segments.csv` ties everything 
together.  
Each row corresponds to one segment-level file and includes segmentation info, split assignment, and SMAD labels.

Paths in the CSV are stored **relative to `data/`**, so the code can reconstruct absolute paths by prefixing `data/`:

```text
raw_file,segment_path,start_time,end_time,split,label_source,speech,music,noise,speech_score,music_score,noise_score
raw/001/001_NO_RAD_0001.wav,segments/001_NO_RAD_0001_s0001.wav,0.0,10.0,dev,ast_pseudo,1,1,0,0.93,0.88,0.05
raw/001/001_NO_RAD_0001.wav,segments/001_NO_RAD_0001_s0002.wav,5.0,15.0,unlabeled,ast_pseudo,0,1,0,0.12,0.91,0.07
raw/001/001_NO_RAD_0002.wav,segments/001_NO_RAD_0002_s0001.wav,0.0,10.0,test,gold,1,0,0,1.00,0.02,0.01
```

## Reproducing experiments

Once you have completed the Installation and Quickstart steps and synced data from Backblaze B2, you can reproduce the main teacher experiments as follows.

1. Run the Audio Spectrogram Transformer teacher over the desired split(s) (for example, `dev` and `test`):
    ```bash
    python -m data_processing.teachers.apply_ast
    ```
    `data/metadata/experiments/` is an optional directory for per-run CSV snapshots. If you prefer, you can keep teacher CSVs directly under `data/metadata/` and only use the versioned HF datasets under `data/metadata/blocs_smad_segments/`.

2. Run the Whisper-AT teacher on the same segments:
    ```bash
    python -m data_processing.teachers.apply_whisper
    ```
    This will write Whisper-AT scores and pseudo-labels (with `label_source = whisper_pseudo`) into the same locations.

3. Run the CLAP and M2D-CLAP zero-shot teachers:
    ```bash
    python -m data_processing.teachers.apply_clap
    python -m data_processing.teachers.apply_m2d
    ```
    These scripts will produce zero-shot labels and scores with `label_source = clap_zero_shot` and `label_source = m2d_zero_shot`, respectively.

4. Open the analysis notebooks to inspect distributions, compare teachers, and (once available) evaluate against gold labels:

    - `notebooks/01_smad_segments_eda.ipynb`
    - `notebooks/02_ast_teacher_sanity.ipynb`
    - `notebooks/03_teacher_comparison.ipynb`

Student fine-tuning under `models/` is under active development; when available, a similar set of commands and configuration files will be documented here for reproducing student domain-adaptation experiments.

## Citation

Coming soon.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
