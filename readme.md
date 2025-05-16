# Airplane Landing Scheduling Project

## Overview
This repository schedules airplane landings using three metaheuristics (Simulated Annealing, Genetic Algorithm, Variable Neighborhood Search) and visualises results with a **Streamlit** dashboard.

```
project_delta/
├── data/landings.csv          # input data
├── results/                   # generated plots & summary CSV
├── src/                       # Python package
│   ├── __init__.py
│   ├── landing_scheduler.py   # penalty + separation checks
│   ├── metaheuristics.py      # SA, GA, VNS classes
│   └── utils.py               # data loading + plotting helpers
├── main.py                    # command‑line driver (runs algorithms + optional dashboard)
├── app.py                     # Streamlit dashboard
├── project_delta.ipynb        # exploratory notebook
├── requirements.txt           # Python deps (pip/conda)
└── README.md                  # this file
```

## Quick start
### 1 · Set up the environment (conda)
```bash
conda create -n landing-sched python=3.9
conda activate landing-sched
# core libs via conda for faster installs
conda install numpy pandas matplotlib notebook tqdm
# remainder via pip
pip install -r requirements.txt
```

### 2 · Run all algorithms
```bash
python main.py --data data/landings.csv --output results
```
*Plots* (`baseline/sa/ga/vns` PNG) and `results_summary.csv` are saved in `results/`.

### 3 · Launch the dashboard
Either separately:
```bash
streamlit run app.py --server.headless true
```
Or automatically after the experiments:
```bash
python main.py --launch-app
```
The script starts Streamlit in headless mode and opens **http://localhost:8501**; press **Ctrl‑C** in the terminal to stop.

---
## Streamlit Dashboard (`app.py`)
* Shows the summary table (Method × Penalty).
* Displays each schedule plot side‑by‑side.
* Sidebar **Refresh Results** button reloads the CSV/plots after re‑running `main.py`.

Column‑name normalisation means the dashboard accepts either lowercase or title‑case headers in `results_summary.csv`.

---
## Algorithm notes
* **Baseline** – greedy sequence sorted by target time, shifted to satisfy the 4‑unit separation.
* **SA / VNS** – simple permutation search using target times as decoding; feel free to extend with repair/shift decoding for tighter penalties.
* **GA** – skips plotting if no feasible individual is found (penalty = ∞).

---
## requirements.txt (pip)
```
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
tqdm>=4.62
streamlit>=1.32
```
(*If you prefer, install `numpy`/`pandas`/`matplotlib` with conda first, then `pip install streamlit tqdm`.*)

---
## Usage reference
```bash
python main.py -h
```
```
--data         path to CSV  (default: data/landings.csv)
--output       results dir  (default: results)
--launch-app   start Streamlit after computations
```

---
## License
MIT

*Happy scheduling & visualising!*
