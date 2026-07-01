"""Neural field and mean field models."""
import numpy as np
from scipy.integrate import solve_ivp


class NeuralFieldModel:
    """1D neural field (Amari-type) with Mexican hat connectivity."""

    def __init__(self, n: int = 100, h: float = -0.1,
                 A_exc: float = 1.0, sigma_exc: float = 5.0,
                 A_inh: float = 0.65, sigma_inh: float = 10.0):
        self.n = n
        self.h = h
        self.x = np.linspace(-50, 50, n)
        dx = self.x[1] - self.x[0]
        dist = np.abs(self.x[:, None] - self.x[None, :])
        self.W = (A_exc * np.exp(-dist**2 / (2 * sigma_exc**2))
                  - A_inh * np.exp(-dist**2 / (2 * sigma_inh**2))) * dx

    def _f(self, u: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        return np.maximum(0, u - threshold)

    def simulate(self, duration: float, dt: float = 1.0,
                 input_fn=None) -> np.ndarray:
        n_steps = int(duration / dt)
        u = np.zeros((n_steps, self.n))
        for t in range(1, n_steps):
            I = input_fn(t * dt, self.x) if input_fn is not None else np.zeros(self.n)
            du = (-u[t-1] + self.W @ self._f(u[t-1]) + self.h + I) * dt
            u[t] = u[t-1] + du
        return u


class MeanFieldModel:
    """Two-population (E/I) mean field with firing-rate dynamics."""

    def __init__(self, J_EE: float = 1.5, J_EI: float = 1.0,
                 J_IE: float = 1.0, J_II: float = 0.5,
                 tau_E: float = 20.0, tau_I: float = 10.0):
        self.J_EE = J_EE
        self.J_EI = J_EI
        self.J_IE = J_IE
        self.J_II = J_II
        self.tau_E = tau_E
        self.tau_I = tau_I

    def _phi(self, x: float, gain: float = 1.0) -> float:
        return gain * np.maximum(0, x)

    def _odes(self, t, y, I_E, I_I):
        r_E, r_I = y
        dr_E = (-r_E + self._phi(self.J_EE * r_E - self.J_EI * r_I + I_E)) / self.tau_E
        dr_I = (-r_I + self._phi(self.J_IE * r_E - self.J_II * r_I + I_I)) / self.tau_I
        return [dr_E, dr_I]

    def simulate(self, duration: float, I_E: float = 1.0, I_I: float = 0.5,
                 dt: float = 0.5) -> np.ndarray:
        t_eval = np.arange(0, duration, dt)
        sol = solve_ivp(self._odes, (0, duration), [0.1, 0.1],
                        args=(I_E, I_I), t_eval=t_eval, method="RK45")
        return sol.y.T
