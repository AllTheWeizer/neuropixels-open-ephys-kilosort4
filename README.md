# Neuropixels Open Ephys → Kilosort4: Reproducible Notebook Pipeline

## Project proposal (WIS Programming in Python using AI)

This project makes an existing Neuropixels processing notebook reproducible and straightforward for lab colleagues to run on a selected Open Ephys recording. It preserves the original **cell-based Jupyter notebook**, adds a validated small test dataset, setup and execution guidance, and will add focused tests and quality checks. This is a continuation and improvement of an existing research pipeline, **not** a claim that the notebook's original analysis was newly written for the course.

## What it does

The main workflow reads a Neuropixels Open Ephys session; optionally crops its start; removes selected bad channels and applies phase correction; filters the spike-band (300 Hz high-pass) and global median references it; saves the exact SpikeInterface recording used for sorting; preserves and checks the original timestamps and sample numbers; then runs **Kilosort4 in Docker**. It records processing settings, sorter inputs and output provenance. Post-sorting curation, LFP and ADC paths remain in the notebook but **are not required for the initial course deliverable**.

### Inputs

- An Open Ephys session containing `structure.oebin`, `continuous/<probe>/continuous.dat`, `timestamps.npy`, `sample_numbers.npy` and `events/`.
- Notebook configuration cell 0.2: `base_path`, recording node/experiment/recording labels, probe folder, initial crop duration, event channel, bad channels and optional ADC mode.
- Local Docker Desktop, NVIDIA GPU exposure via WSL 2, and a Kilosort4 Docker image to execute S4 (a software prerequisite, not a data file).

A **12.82-second example** can be downloaded separately as `2026-0822_473nm_optotest_cosine.zip`. It includes 384-channel 30-kHz Neuropixels data and NI-DAQ data at 40 kHz. The recording is for fast **I/O and preprocessing checks**, not meaningful spike-sort quality validation.

### Outputs

- Saved SpikeInterface `si_preprocessed/` spike-band recording and matching preprocessing-settings JSON.
- `timestamps_trimmed.npy`, `sample_numbers_trimmed.npy`, `timestamp_gap_report.json` and provenance metadata.
- `kilosort4/sorter_output/` (spike times, clusters and additional sorter outputs) plus `nam_kilosort_run_record.json`, if S4 completes.
- Optional event, LFP and curation files from the remaining notebook sections.

## Download, install and run (Windows / VS Code)

1. Clone this repository, or unzip it into a short directory such as `C:\np-course`. Open the **folder** in VS Code.
2. **Do not commit raw recordings.** Place the original data ZIP outside the Git repository or under `test_data/` (ignored by Git). Extract it so the folder structure is:

   ```text
   test_data/2026-0822_473nm_optotest_cosine/
       2026-08-22_02-27-46/
           Record Node 112/experiment1/recording1/
               structure.oebin
               continuous/Neuropix-PXI-104.ProbeA/continuous.dat
               continuous/Neuropix-PXI-104.ProbeA/timestamps.npy
               continuous/Neuropix-PXI-104.ProbeA/sample_numbers.npy
               events/...
   ```

3. Create a virtual environment and install Python dependencies (activation is **not required**):

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Check the ZIP before extraction (optional but recommended):

   ```powershell
   .\.venv\Scripts\python.exe verify_test_data.py C:\path\to\2026-0822_473nm_optotest_cosine.zip
   ```

5. In VS Code, open `Neuropixels_OpenEphys_to_Kilosort4.ipynb`, select the `.venv` Python kernel, and run **individual cells in sequence**:

   `0.1 → 0.2 → 1.1 → S1 → S2 → S3 → S4`

   Keep **`REMOVE_FIRST_MINUTES = 0`** for the 12.82-second example. The prepared notebook's only essential project-specific configuration edit is `base_path = Path.cwd() / 'test_data' / ...`. The default `ADC_MODE = 'off'` leaves the included NI-DAQ data unused; opt in later if you want to validate analog synchronization.

   Do **not** use Jupyter's *Run All*: later notebook cells include optional curation, LFP and alignment branches with separate prerequisites. A final legacy debugging cell can be ignored.

6. To run S4, start Docker Desktop and verify GPU availability. The notebook expects this pinned image:

   ```powershell
   docker pull spikeinterface/kilosort4-base:4.0.38_cuda-12.0.0
   docker run --rm --gpus all spikeinterface/kilosort4-base:4.0.38_cuda-12.0.0 nvidia-smi
   ```

   S4 will stop if the Docker image or GPU is unavailable. A successful input validation or S1–S3 run must **not** be represented as successful Kilosort execution.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe verify_test_data.py C:\path\to\2026-0822_473nm_optotest_cosine.zip
```

The pytest tests check the archive-validator's logic using generated tiny fake recordings. The real archive preflight verifies the presence and consistency of channels, raw samples and timestamps. Full preprocessing and S4 require your lab environment and must be tested there; automatic tests in this repository do **not** prove that Kilosort sorted the short example successfully.

## Course project development goals

The baseline is an **existing research notebook**. New course work will focus on improving the reproducibility and usability of the existing workflow rather than rewriting its scientific algorithms: (1) portable configuration and archive preflight; (2) explicit setup and error messages; (3) repeatable execution on a small test recording and saved QC/provenance checks; (4) additional automated tests and a recorded demonstration of S1–S4, subject to the available GPU/Docker hardware. Major changes to the underlying algorithms are out of scope without separate scientific validation.

**Proposal status:** prepared for instructor approval; full pipeline execution on your own workstation is not yet verified.

Developed as a project for the [Weizmann Programming in Python using AI course](https://github.com/Code-Maven/wis-python-course-2026-10).
