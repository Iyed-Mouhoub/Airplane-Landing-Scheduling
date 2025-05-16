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
def repair_schedule(order: List[int],
                    earliest: np.ndarray,
                    target: np.ndarray,
                    separation: float = 4.0
                   ) -> (np.ndarray, np.ndarray):
    """
    Given an order, build a feasible landing schedule by shifting
    each flight forward to satisfy its time window and the minimum
    separation. Returns (pos_times, times_by_id):
      - pos_times[pos] = scheduled time of the flight in that landing position
      - times_by_id[ac_id] = scheduled time of flight with that ID
    """
    n = len(order)
    pos_times = np.zeros(n)
    for pos, ac in enumerate(order):
        t_pref, t_ear = target[ac], earliest[ac]
        if pos == 0:
            pos_times[pos] = max(t_ear, t_pref)
        else:
            pos_times[pos] = max(t_ear, t_pref, pos_times[pos-1] + separation)

    times_by_id = np.zeros(n)
    for pos, ac in enumerate(order):
        times_by_id[ac] = pos_times[pos]

    return pos_times, times_by_id
