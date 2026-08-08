# Dynamic Difficulty Adjustment in a Virtual Reality Game: Comparing Fixed Progression with Two Adaptive Progression Strategies

This repository contains the dataset and Python analysis code associated with the study of difficulty progression in *Pulse Run*, an arcade-style virtual reality game.

The study compares three progression strategies:

- Fixed Time-Based Progression
- Performance-Based Dynamic Difficulty Adjustment (DDA)
- Skill-Specific DDA

The analysis examines delivered pattern difficulty, gameplay performance, and post-run player responses. The primary analysis includes sessions completed on Meta Quest 3. A sensitivity analysis includes all eligible Meta Quest 2 and Meta Quest 3 sessions.

## Repository Contents

- `Pulse_Run_Section_4_Analysis.ipynb`: main Jupyter notebook.
- `run_analysis.py`: command-line entry point for the complete analysis.
- `pulse_run_analysis/`: data processing, statistical analysis, reporting, and plotting code.
- `analysis_config.toml`: analysis paths, screening criteria, and statistical settings.
- `data/raw/`: study data in JSON format.
- `review/`: source and session review metadata used during screening.
- `outputs/`: processed datasets, statistical results, tables, figures, and diagnostics.
- `requirements.txt`: required Python packages.

## Requirements

- Python 3.11 or newer
- Packages listed in `requirements.txt`

## Running the Analysis

Create and activate a virtual environment, then install the required packages.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the complete analysis from the repository root:

```bash
python run_analysis.py
```

To use the notebook instead:

```bash
python -m jupyter lab
```

Open `Pulse_Run_Section_4_Analysis.ipynb` and run all cells. Generated files are written to `outputs/`.

## Data

The study data include age, previous VR experience, gameplay telemetry, progression-strategy assignment, and brief post-run responses. No names, contact details, audio recordings, or video recordings were collected.

## Citation

If you use this dataset or analysis code, please cite the associated paper:

> *Dynamic Difficulty Adjustment in a Virtual Reality Game: Comparing Fixed Progression with Two Adaptive Progression Strategies.*

The complete citation will be added after publication.
