"""Phase portrait and state space utilities."""
from typing import Callable, Tuple

import numpy as np


def nullclines_2d(f: Callable, x_range: Tuple, y_range: Tuple,
                  resolution: int = 50):
    """Compute nullclines for a 2D ODE system."""
    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(xs, ys)
    dX = np.zeros_like(X)
    dY = np.zeros_like(Y)
    for i in range(resolution):
        for j in range(resolution):
            dy = f(0, [X[i, j], Y[i, j]])
            dX[i, j] = dy[0]
            dY[i, j] = dy[1]
    return X, Y, dX, dY


def phase_trajectory(f: Callable, y0: np.ndarray, duration: float,
                     dt: float = 0.01) -> np.ndarray:
    """Simple Euler integration for phase portrait."""
    from scipy.integrate import solve_ivp
    sol = solve_ivp(f, (0, duration), y0, max_step=dt, method="RK45")
    return sol.y.T


def find_fixed_points(f: Callable, y0_list: list,
                      tol: float = 1e-6) -> list:
    """Find fixed points via Newton iteration from multiple starting points."""
    from scipy.optimize import fsolve
    fps = []
    for y0 in y0_list:
        try:
            fp = fsolve(lambda y: f(0, y), y0, full_output=True)
            if fp[2] == 1:  # converged
                fps.append(fp[0])
        except Exception:
            pass
    return fps
