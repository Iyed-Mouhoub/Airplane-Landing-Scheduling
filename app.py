# app.py
import os
import streamlit as st
import pandas as pd

# Constants
RESULTS_DIR = 'results'
SUMMARY_CSV = os.path.join(RESULTS_DIR, 'results_summary.csv')
PLOTS = {
    'Baseline': 'baseline_schedule.png',
    'Simulated Annealing': 'sa_schedule.png',
    'Genetic Algorithm': 'ga_schedule.png',
    'Variable Neighborhood Search': 'vns_schedule.png'
}

st.set_page_config(page_title='Airplane Landing Scheduling', layout='wide')

st.title('🛬 Airplane Landing Scheduling Dashboard')

st.markdown(
    """
    This app displays the results of different metaheuristic algorithms applied to the airplane landing scheduling problem.

    - **Baseline** – Feasible greedy schedule
    - **Simulated Annealing (SA)**
    - **Genetic Algorithm (GA)**
    - **Variable Neighborhood Search (VNS)**
    """
)

# ---------------------------------------------------------------------
# Helper: load summary CSV and standardise column names               
# ---------------------------------------------------------------------
@st.cache_data
def load_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalise column name capitalisation so either "method" or "Method" works
    df.columns = [c.strip().title() for c in df.columns]
    return df

summary_df = load_summary(SUMMARY_CSV)

# ---------------------------------------------------------------------
# Display area                                                         
# ---------------------------------------------------------------------
expected_cols = {'Method', 'Penalty'}

if summary_df.empty:
    st.warning('No results found. Run `main.py` first to generate results.')
elif not expected_cols.issubset(summary_df.columns):
    st.error(f'`results_summary.csv` must contain columns: {expected_cols}. Found {list(summary_df.columns)}')
else:
    # Results table ---------------------------------------------------
    st.subheader('Results Summary')
    st.dataframe(summary_df.set_index('Method'))

    # Schedule plots --------------------------------------------------
    st.subheader('Schedules')
    cols = st.columns(2)
    for idx, row in summary_df.iterrows():
        method = row['Method']
        col = cols[idx % 2]
        plot_path = os.path.join(RESULTS_DIR, PLOTS.get(method, ''))
        if os.path.exists(plot_path):
            col.markdown(f'**{method}**')
            col.image(plot_path, use_container_width=True)

        else:
            col.markdown(f'**{method}** – plot not found')

# Sidebar controls ----------------------------------------------------
st.sidebar.title('Controls')
if st.sidebar.button('Refresh Results', key='refresh'):
    st.experimental_rerun()
