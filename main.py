# main.py
import argparse
import os
import subprocess
import webbrowser
import numpy as np
import pandas as pd

from src.utils import load_data, plot_schedule
from src.landing_scheduler import compute_penalty, check_separation
from src.metaheuristics import (
    SimulatedAnnealing,
    GeneticAlgorithm,
    VariableNeighborhoodSearch,
)

# Ensure matplotlib uses non‑GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEP = 4.0  # global separation constant

# -------------------------------------------------------------------
# Baseline
# -------------------------------------------------------------------

def run_baseline(data, out_dir):
    n = len(data["target"])
    order = sorted(range(n), key=lambda i: data["target"][i])
    pos_times = np.zeros(n)
    for pos, ac in enumerate(order):
        t_pref, t_ear = data["target"][ac], data["earliest"][ac]
        pos_times[pos] = (
            max(t_ear, t_pref)
            if pos == 0
            else max(t_ear, t_pref, pos_times[pos - 1] + SEP)
        )
    times_by_id = np.zeros(n)
    for pos, ac in enumerate(order):
        times_by_id[ac] = pos_times[pos]

    penalty = compute_penalty(times_by_id, **data)
    path = os.path.join(out_dir, "baseline_schedule.png")
    plot_schedule(order, pos_times, outfile=path)
    plt.close("all")
    print(f"Baseline penalty: {penalty:.2f}")
    print(f"Saved baseline schedule plot to {path}")
    return {"Method": "Baseline", "Penalty": penalty}

# -------------------------------------------------------------------
# SA
# -------------------------------------------------------------------

def run_sa(data, out_dir):
    sa = SimulatedAnnealing(data, T0=1000, alpha=0.98, iter_per_temp=200)
    order, pen = sa.run()
    path = os.path.join(out_dir, "sa_schedule.png")
    plot_schedule(order, [data["target"][i] for i in order], outfile=path)
    plt.close("all")
    print(f"SA best penalty: {pen:.2f}")
    print(f"Saved SA schedule plot to {path}")
    return {"Method": "Simulated Annealing", "Penalty": pen}

# -------------------------------------------------------------------
# GA
# -------------------------------------------------------------------

def run_ga(data, out_dir):
    ga = GeneticAlgorithm(data, pop_size=60, cx_rate=0.8, mut_rate=0.2, generations=100)
    order, pen = ga.run()
    if order is not None and np.isfinite(pen):
        path = os.path.join(out_dir, "ga_schedule.png")
        plot_schedule(order, [data["target"][i] for i in order], outfile=path)
        plt.close("all")
        print(f"Saved GA schedule plot to {path}")
    else:
        print("GA did not find a feasible solution – skipping plot.")
    print(f"GA best penalty: {pen}")
    return {"Method": "Genetic Algorithm", "Penalty": pen}

# -------------------------------------------------------------------
# VNS
# -------------------------------------------------------------------

def run_vns(data, out_dir):
    vns = VariableNeighborhoodSearch(data, max_iter=1000)
    order, pen = vns.run()
    path = os.path.join(out_dir, "vns_schedule.png")
    plot_schedule(order, [data["target"][i] for i in order], outfile=path)
    plt.close("all")
    print(f"VNS best penalty: {pen:.2f}")
    print(f"Saved VNS schedule plot to {path}")
    return {"Method": "Variable Neighborhood Search", "Penalty": pen}

# -------------------------------------------------------------------
# Launch Streamlit helper
# -------------------------------------------------------------------

def launch_streamlit():
    """Launch the Streamlit dashboard and wait until the user stops it (Ctrl‑C)."""
    import shutil

    if shutil.which("streamlit") is None:
        print("Streamlit is not installed. Please run `pip install streamlit` and try again.")
        return

    print("Starting Streamlit dashboard on http://localhost:8501 …  (Press Ctrl‑C to stop)")
    proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            "--server.port=8501",
        ]
    )
    try:
        webbrowser.open("http://localhost:8501", new=1, autoraise=True)
    except Exception:
        pass

    # Wait for Streamlit process so main.py doesn't exit immediately
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("Stopping Streamlit …")
        proc.terminate()
        proc.wait()
# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run landing scheduling experiments and optionally launch Streamlit dashboard.")
    parser.add_argument("--data", default="data/landings.csv", help="Path to CSV data file")
    parser.add_argument("--output", default="results", help="Directory to store results")
    parser.add_argument("--launch-app", action="store_true", help="Launch Streamlit app after finishing experiments")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data = load_data(args.data)

    results = [
        run_baseline(data, args.output),
        run_sa(data, args.output),
        run_ga(data, args.output),
        run_vns(data, args.output),
    ]

    # Save summary CSV
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(args.output, "results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary results to {summary_path}")

    if args.launch_app:
        launch_streamlit()


if __name__ == "__main__":
    main()
