"""Keep ByteTrack working when Windows blocks the `lap` / `lapx` compiled file.

Ultralytics' tracker solves "which detection belongs to which track" with lap.lapjv from the `lapx` package. On some
Windows PCs, Application Control / Smart App Control blocks that package's DLL (_lapjv), and tracking cannot start.
install_lap_shim() checks whether `lap` loads; if it does not, it registers a drop-in replacement for lapjv built on
SciPy's linear_sum_assignment (the solver Ultralytics itself falls back to). Results match lapjv except in rare ties.

Call it BEFORE importing ultralytics. It does nothing when the real `lap` works.
"""
import sys
import types

import numpy as np


def _lapjv(cost, extend_cost=False, cost_limit=np.inf, return_cost=True):
    """Same return shape as lap.lapjv: (total_cost, x, y); x[row] = column or -1, y[column] = row or -1."""
    from scipy.optimize import linear_sum_assignment
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    x = np.full(n, -1, dtype=np.int64)
    y = np.full(m, -1, dtype=np.int64)
    total = 0.0
    if n and m:
        rows, cols = linear_sum_assignment(np.nan_to_num(cost, nan=1e6, posinf=1e6))
        for r, c in zip(rows, cols):
            if cost[r, c] <= cost_limit:                 # pairs costing more than the limit stay unmatched
                x[r], y[c] = c, r
                total += float(cost[r, c])
    return (total, x, y) if return_cost else (x, y)


def install_lap_shim():
    """Returns True if the shim was installed (the real `lap` could not be loaded), else False."""
    try:
        import lap                                       # importing can succeed while the DLL is still blocked...
        lap.lapjv(np.array([[0.1, 0.9], [0.8, 0.2]]), extend_cost=True, cost_limit=0.5)   # ...so really call it once
        return False
    except Exception:
        pass
    try:
        import scipy.optimize                            # noqa: F401  (needed by the replacement)
    except Exception as e:
        print(f"[TRACKER] lap is blocked and SciPy is missing too ({e}); tracking will fail. Fix: pip install scipy")
        return False
    sys.modules.pop("lap", None)
    mod = types.ModuleType("lap")
    mod.__version__ = "0.5.12+safesight-shim"
    mod.lapjv = _lapjv
    sys.modules["lap"] = mod
    print("[TRACKER] 'lap' could not be loaded (blocked by Windows?); using the SciPy-based replacement.")
    return True