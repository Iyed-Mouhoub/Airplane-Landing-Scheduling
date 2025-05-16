import os
import pandas as pd
import streamlit as st

# Constants
data_dir = "data"
SUMMARY_CSV = os.path.join("results", "results_summary.csv")
PLOTS_DIR = "results"

# Title & description
st.set_page_config(layout="wide")
st.title("✈️ Airplane Landing Scheduling Dashboard")
st.markdown(
    """
    This app displays the results of different metaheuristic algorithms 
    applied to the airplane landing scheduling problem.

    - **Baseline**: Feasible greedy schedule
    - **Simulated Annealing (SA)**
    - **Genetic Algorithm (GA)**
    - **Variable Neighborhood Search (VNS)**
    """
)

# Controls
with st.sidebar:
    if st.button("Refresh Results", key="refresh"):
        st.experimental_rerun()

# Summary loader (re-reading on each run)
@st.cache_data
def load_summary(path: str) -> pd.DataFrame:
    """
    Load the summary CSV. Cached, but invalidated automatically when the file changes.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().title() for c in df.columns]
    return df

# Load and display summary
summary_df = load_summary(SUMMARY_CSV)
if summary_df.empty:
    st.warning("No results found. Run `main.py` first to generate results.")
else:
    st.subheader("Results Summary")
    st.dataframe(summary_df.set_index('Method'))

    # Display schedule plots
    st.subheader("Schedules")
    cols = st.columns(2)

    # map method names to generated plot filenames
    filename_map = {
        'Baseline': 'baseline_schedule.png',
        'Simulated Annealing': 'sa_schedule.png',
        'Genetic Algorithm': 'ga_schedule.png',
        'Variable Neighborhood Search': 'vns_schedule.png'
    }

    for idx, (_, row) in enumerate(summary_df.iterrows()):
        method = row['Method']
        plot_file = filename_map.get(method, '')
        plot_path = os.path.join(PLOTS_DIR, plot_file)
        col = cols[idx % 2]
        col.markdown(f"**{method}**")
        if plot_file and os.path.exists(plot_path):
            col.image(plot_path, use_container_width=True)
        else:
            col.info(f"Plot for {method} not found.")

    # Optionally: add convergence curves here
    # ...
