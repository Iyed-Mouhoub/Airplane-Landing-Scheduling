# utils.py

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

def load_data(filepath: str) -> Dict[str, Any]:
    """
    Load landing data CSV with columns:
      earliest, target, latest, earliness_penalty, lateness_penalty
    Returns a dict of numpy arrays.
    """
    df = pd.read_csv(filepath)
    return {
        'earliest':          df['earliest'].to_numpy(),
        'target':            df['target'].to_numpy(),
        'latest':            df['latest'].to_numpy(),
        'alpha':             df['earliness_penalty'].to_numpy(),
        'beta':              df['lateness_penalty'].to_numpy(),
    }

# src/utils.py

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

def plot_schedule(order: List[int],
                  times: List[float],
                  outfile: Optional[str] = None) -> None:
    """
    Gantt-style plot of landing schedule.
    - order: sequence of aircraft indices
    - times: landing times corresponding to order positions
    - outfile: if provided, path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, max(4, len(order)*0.3)))
    for pos, idx in enumerate(order):
        start = times[pos]
        ax.broken_barh([(start, 0.5)], (pos-0.2, 0.4), facecolor='tab:blue')
        ax.text(start + 0.1, pos, f"AC{idx+1}", va='center')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"Pos {i+1}" for i in range(len(order))])
    ax.set_xlabel("Time")
    ax.set_title("Landing Schedule")
    plt.tight_layout()

    if outfile:
        fig.savefig(outfile)
    # No plt.show() when using Agg backend
    plt.close(fig)

    