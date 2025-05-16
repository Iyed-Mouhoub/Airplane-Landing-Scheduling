# Airplane Landing Scheduling Project

## Overview

This repository implements scheduling of airplane landings using metaheuristic methods (Simulated Annealing, Genetic Algorithm, and Variable Neighborhood Search). It includes:

- **Source code** in `src/`:
  - `landing_scheduler.py` — penalty calculation and separation checks
  - `metaheuristics.py` — implementations of SA, GA, and VNS
  - `utils.py` — data loading and schedule plotting utilities
- **Jupyter Notebook** `project_delta.ipynb` — interactive workflow:
  1. Data loading
  2. Baseline heuristic
  3. SA, GA, and VNS runs
  4. Comparison of results
  5. (Optional) convergence plots
- **Data folder** `data/landings.csv` — input data file with columns:
  - `earliest`, `target`, `latest`, `earliness_penalty`, `lateness_penalty`
- **`main.py`** — script to run all experiments and save outputs
- **`requirements.txt`** — Python dependencies
- **README.md** — this file

---

## Project Structure

```
project_delta/
├── data/
│   └── landings.csv
├── results/            # output plots and summary CSV
├── src/
│   ├── __init__.py
│   ├── landing_scheduler.py
│   ├── metaheuristics.py
│   └── utils.py
├── project_delta.ipynb
├── main.py
├── requirements.txt
└── README.md
```

---

## Setup with Conda

1. **Create and activate** a Conda environment:
   ```bash
   conda create -n landing-sched python=3.9
   conda activate landing-sched
   ```

2. **Install core packages via Conda**:
   ```bash
   conda install numpy pandas matplotlib notebook tqdm
   ```

3. **Install remaining dependencies** from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify** installation:
   ```bash
   conda list
   ```

---

## Usage

### 1. Jupyter Notebook

Launch Jupyter and open the notebook:
```bash
jupyter notebook project_delta.ipynb
```

Run each cell sequentially. The notebook will:
- Load and display the data
- Compute and plot a baseline schedule
- Execute SA, GA, and VNS, plotting each result
- Show a comparison table of penalties
- (Optionally) plot convergence curves

### 2. Command-Line Script

Run all experiments with:
```bash
python main.py --data data/landings.csv --output results
```
This will:
- Compute baseline, SA, GA, and VNS schedules
- Save each schedule plot in `results/`
- Generate `results_summary.csv` with method vs. penalty

---

## Customization

- **Algorithm parameters**: adjust defaults in `src/metaheuristics.py` or pass arguments when instantiating in `main.py`.
- **Separation time**: modify the default `4.0` in `src/landing_scheduler.py`.
- **Convergence logging**: extend classes to record history arrays for plotting.

---

## Exporting the Report

To generate a PDF from the notebook:
```bash
jupyter nbconvert --to pdf project_delta.ipynb
```
Or Markdown:
```bash
jupyter nbconvert --to markdown project_delta.ipynb
```

---

## License

MIT License

---

*Happy scheduling!*
