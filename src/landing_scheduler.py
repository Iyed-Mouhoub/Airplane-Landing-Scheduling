# landing_scheduler.py

import numpy as np
from typing import List

def compute_penalty(times: np.ndarray,
                    earliest: np.ndarray,
                    target: np.ndarray,
                    latest: np.ndarray,
                    alpha: np.ndarray,
                    beta: np.ndarray) -> float:
    """
    Compute total earliness and lateness penalty for a schedule.
    - times: scheduled landing times (n,)
    - earliest, target, latest: window bounds (n,)
    - alpha, beta: per-unit earliness/lateness penalties (n,)
    """
    earliness = np.maximum(earliest - times, 0)
    lateness  = np.maximum(times - target,   0)
    return float(np.sum(alpha * earliness + beta * lateness))

def check_separation(order: List[int],
                     times: np.ndarray,
                     separation: float = 4.0) -> bool:
    """
    Verify that in the given order, each consecutive pair respects the minimum separation.
    - order: list of aircraft indices in landing sequence
    - times: scheduled landing times (n,)
    - separation: minimum required gap (default = 4.0)
    """
    for i in range(len(order) - 1):
        a, b = order[i], order[i+1]
        if times[b] - times[a] < separation:
            return False
    return True
