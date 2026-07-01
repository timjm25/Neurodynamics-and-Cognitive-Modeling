from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class SimResult:
    t: np.ndarray
    V: np.ndarray
    spike_times: np.ndarray
    state_vars: Dict[str, np.ndarray] = field(default_factory=dict)


class LeakyIntegrateAndFire:
    name = "lif"
    description = "Leaky Integrate-and-Fire neuron model"

    def __init__(self, tau_m: float = 20.0, v_rest: float = -70.0,
                 v_thresh: float = -55.0, v_reset: float = -75.0,
                 r_m: float = 10.0):
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.r_m = r_m

    def simulate(self, duration: float, current: float = 0.0,
                 dt: float = 0.1, current_fn=None) -> SimResult:
        n_steps = int(duration / dt)
        t = np.arange(n_steps) * dt
        V = np.zeros(n_steps)
        V[0] = self.v_rest
        spike_times = []

        for i in range(1, n_steps):
            I = current_fn(t[i - 1]) if current_fn is not None else current
            dV = (-(V[i - 1] - self.v_rest) + self.r_m * I) / self.tau_m
            V[i] = V[i - 1] + dt * dV
            if V[i] >= self.v_thresh:
                spike_times.append(t[i])
                V[i] = self.v_reset

        return SimResult(t=t, V=V, spike_times=np.array(spike_times))

    def step(self, inputs: float, state: Dict) -> Dict:
        V = state.get("V", self.v_rest)
        dt = state.get("dt", 0.1)
        dV = (-(V - self.v_rest) + self.r_m * inputs) / self.tau_m
        V_new = V + dt * dV
        spiked = V_new >= self.v_thresh
        if spiked:
            V_new = self.v_reset
        return {"V": V_new, "spiked": spiked, "dt": dt}


class AdaptiveExponential:
    name = "adex"
    description = "Adaptive Exponential Integrate-and-Fire"

    def __init__(self, C: float = 281.0, g_L: float = 30.0, E_L: float = -70.6,
                 V_T: float = -50.4, delta_T: float = 2.0, tau_w: float = 144.0,
                 a: float = 4.0, b: float = 80.5, V_r: float = -70.6,
                 V_peak: float = 20.0):
        self.C = C
        self.g_L = g_L
        self.E_L = E_L
        self.V_T = V_T
        self.delta_T = delta_T
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.V_r = V_r
        self.V_peak = V_peak

    def simulate(self, duration: float, current: float = 500.0,
                 dt: float = 0.05) -> SimResult:
        n_steps = int(duration / dt)
        t = np.arange(n_steps) * dt
        V = np.zeros(n_steps)
        w = np.zeros(n_steps)
        V[0] = self.E_L
        w[0] = 0.0
        spike_times = []

        for i in range(1, n_steps):
            V_prev, w_prev = V[i - 1], w[i - 1]
            exp_term = self.g_L * self.delta_T * np.exp(
                np.clip((V_prev - self.V_T) / self.delta_T, -20, 20))
            dV = (-self.g_L * (V_prev - self.E_L) + exp_term - w_prev + current) / self.C
            dw = (self.a * (V_prev - self.E_L) - w_prev) / self.tau_w
            V[i] = V_prev + dt * dV
            w[i] = w_prev + dt * dw
            if V[i] >= self.V_peak:
                spike_times.append(t[i])
                V[i] = self.V_r
                w[i] = w[i] + self.b

        return SimResult(t=t, V=V, spike_times=np.array(spike_times),
                         state_vars={"w": w})

    def step(self, inputs: float, state: Dict) -> Dict:
        V = state.get("V", self.E_L)
        w = state.get("w", 0.0)
        dt = state.get("dt", 0.05)
        exp_term = self.g_L * self.delta_T * np.exp(
            np.clip((V - self.V_T) / self.delta_T, -20, 20))
        dV = (-self.g_L * (V - self.E_L) + exp_term - w + inputs) / self.C
        dw = (self.a * (V - self.E_L) - w) / self.tau_w
        V_new = V + dt * dV
        w_new = w + dt * dw
        spiked = V_new >= self.V_peak
        if spiked:
            V_new = self.V_r
            w_new = w_new + self.b
        return {"V": V_new, "w": w_new, "spiked": spiked, "dt": dt}


def _hh_alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10) + 1e-9)
def _hh_beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)
def _hh_alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def _hh_beta_h(V): return 1.0 / (1 + np.exp(-(V + 35) / 10))
def _hh_alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10) + 1e-9)
def _hh_beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)


class HodgkinHuxley:
    name = "hodgkin_huxley"
    description = "Hodgkin-Huxley conductance-based neuron"

    def __init__(self, C_m: float = 1.0, g_Na: float = 120.0, g_K: float = 36.0,
                 g_L: float = 0.3, E_Na: float = 50.0, E_K: float = -77.0,
                 E_L: float = -54.387):
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L

    def _odes(self, t, y, I_ext):
        V, m, h, n = y
        dV = (I_ext - self.g_Na * m**3 * h * (V - self.E_Na)
              - self.g_K * n**4 * (V - self.E_K)
              - self.g_L * (V - self.E_L)) / self.C_m
        dm = _hh_alpha_m(V) * (1 - m) - _hh_beta_m(V) * m
        dh = _hh_alpha_h(V) * (1 - h) - _hh_beta_h(V) * h
        dn = _hh_alpha_n(V) * (1 - n) - _hh_beta_n(V) * n
        return [dV, dm, dh, dn]

    def simulate(self, duration: float, current: float = 10.0,
                 dt: float = 0.025) -> SimResult:
        V0 = -65.0
        m0 = _hh_alpha_m(V0) / (_hh_alpha_m(V0) + _hh_beta_m(V0))
        h0 = _hh_alpha_h(V0) / (_hh_alpha_h(V0) + _hh_beta_h(V0))
        n0 = _hh_alpha_n(V0) / (_hh_alpha_n(V0) + _hh_beta_n(V0))
        y0 = [V0, m0, h0, n0]
        t_span = (0, duration)
        t_eval = np.arange(0, duration, dt)
        sol = solve_ivp(self._odes, t_span, y0, args=(current,),
                        t_eval=t_eval, method="RK45", max_step=dt)
        V = sol.y[0]
        spike_times = t_eval[np.where((V[:-1] < 0) & (V[1:] >= 0))[0]]
        return SimResult(t=sol.t, V=V, spike_times=spike_times,
                         state_vars={"m": sol.y[1], "h": sol.y[2], "n": sol.y[3]})

    def step(self, inputs: float, state: Dict) -> Dict:
        V = state.get("V", -65.0)
        m = state.get("m", 0.05)
        h = state.get("h", 0.6)
        n = state.get("n", 0.32)
        dt = state.get("dt", 0.025)
        dy = self._odes(0, [V, m, h, n], inputs)
        return {"V": V + dt * dy[0], "m": m + dt * dy[1],
                "h": h + dt * dy[2], "n": n + dt * dy[3],
                "spiked": False, "dt": dt}


class Izhikevich:
    name = "izhikevich"
    description = "Izhikevich spiking neuron"

    def __init__(self, a: float = 0.02, b: float = 0.2,
                 c: float = -65.0, d: float = 8.0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def simulate(self, duration: float, current: float = 10.0,
                 dt: float = 0.25) -> SimResult:
        n_steps = int(duration / dt)
        t = np.arange(n_steps) * dt
        v = np.zeros(n_steps)
        u = np.zeros(n_steps)
        v[0] = -65.0
        u[0] = self.b * v[0]
        spike_times = []

        for i in range(1, n_steps):
            dv = (0.04 * v[i-1]**2 + 5 * v[i-1] + 140 - u[i-1] + current)
            du = self.a * (self.b * v[i-1] - u[i-1])
            v[i] = v[i-1] + dt * dv
            u[i] = u[i-1] + dt * du
            if v[i] >= 30:
                spike_times.append(t[i])
                v[i] = self.c
                u[i] = u[i] + self.d

        return SimResult(t=t, V=v, spike_times=np.array(spike_times),
                         state_vars={"u": u})

    def step(self, inputs: float, state: Dict) -> Dict:
        v = state.get("v", -65.0)
        u = state.get("u", self.b * -65.0)
        dt = state.get("dt", 0.25)
        dv = 0.04 * v**2 + 5 * v + 140 - u + inputs
        du = self.a * (self.b * v - u)
        v_new = v + dt * dv
        u_new = u + dt * du
        spiked = v_new >= 30
        if spiked:
            v_new = self.c
            u_new = u_new + self.d
        return {"v": v_new, "u": u_new, "V": v_new, "spiked": spiked, "dt": dt}


class FitzHughNagumo:
    name = "fitzhugh_nagumo"
    description = "FitzHugh-Nagumo excitable system"

    def __init__(self, a: float = 0.7, b: float = 0.8, tau: float = 12.5,
                 I_ext: float = 0.5):
        self.a = a
        self.b = b
        self.tau = tau
        self.I_ext = I_ext

    def _odes(self, t, y, I):
        V, W = y
        dV = V - V**3 / 3 - W + I
        dW = (V + self.a - self.b * W) / self.tau
        return [dV, dW]

    def simulate(self, duration: float, current: Optional[float] = None,
                 dt: float = 0.1) -> SimResult:
        I = current if current is not None else self.I_ext
        t_eval = np.arange(0, duration, dt)
        sol = solve_ivp(self._odes, (0, duration), [-1.0, -0.5],
                        args=(I,), t_eval=t_eval, method="RK45")
        V = sol.y[0]
        dV = np.diff(V)
        peaks = np.where((dV[:-1] > 0) & (dV[1:] <= 0) & (V[1:-1] > 0.5))[0]
        return SimResult(t=sol.t, V=V, spike_times=sol.t[peaks + 1],
                         state_vars={"W": sol.y[1]})

    def step(self, inputs: float, state: Dict) -> Dict:
        V = state.get("V", -1.0)
        W = state.get("W", -0.5)
        dt = state.get("dt", 0.1)
        dy = self._odes(0, [V, W], inputs)
        return {"V": V + dt * dy[0], "W": W + dt * dy[1],
                "spiked": False, "dt": dt}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class WilsonCowan:
    name = "wilson_cowan"
    description = "Wilson-Cowan excitatory-inhibitory population model"

    def __init__(self, tau_e: float = 10.0, tau_i: float = 10.0,
                 w_ee: float = 10.0, w_ei: float = 12.0,
                 w_ie: float = 10.0, w_ii: float = 3.0,
                 r_e: float = 1.0, r_i: float = 1.0,
                 P: float = 1.0, Q: float = 1.0):
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.w_ee = w_ee
        self.w_ei = w_ei
        self.w_ie = w_ie
        self.w_ii = w_ii
        self.r_e = r_e
        self.r_i = r_i
        self.P = P
        self.Q = Q

    def _odes(self, t, y):
        E, I = y
        dE = (-E + (1 - self.r_e * E) * _sigmoid(
            self.w_ee * E - self.w_ei * I + self.P)) / self.tau_e
        dI = (-I + (1 - self.r_i * I) * _sigmoid(
            self.w_ie * E - self.w_ii * I + self.Q)) / self.tau_i
        return [dE, dI]

    def simulate(self, duration: float, dt: float = 0.5,
                 E0: float = 0.1, I0: float = 0.05) -> SimResult:
        t_eval = np.arange(0, duration, dt)
        sol = solve_ivp(self._odes, (0, duration), [E0, I0],
                        t_eval=t_eval, method="RK45")
        return SimResult(t=sol.t, V=sol.y[0],
                         spike_times=np.array([]),
                         state_vars={"E": sol.y[0], "I": sol.y[1]})

    def step(self, inputs: float, state: Dict) -> Dict:
        E = state.get("E", 0.1)
        I = state.get("I", 0.05)
        dt = state.get("dt", 0.5)
        dy = self._odes(0, [E, I])
        return {"E": E + dt * dy[0], "I": I + dt * dy[1], "V": E + dt * dy[0],
                "spiked": False, "dt": dt}


class NeuralMassModel:
    """Simplified Jansen-Rit neural mass model."""
    name = "neural_mass"
    description = "Jansen-Rit neural mass model (pyramidal + excitatory + inhibitory)"

    def __init__(self, A: float = 3.25, B: float = 22.0, a: float = 100.0,
                 b: float = 50.0, v0: float = 6.0, e0: float = 2.5,
                 r: float = 0.56, C: float = 135.0):
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.v0 = v0
        self.e0 = e0
        self.r = r
        self.C = C

    def _sigm(self, v: float) -> float:
        return 2 * self.e0 / (1 + np.exp(self.r * (self.v0 - v)))

    def simulate(self, duration: float, dt: float = 1.0,
                 p_mean: float = 220.0, p_std: float = 22.0,
                 seed: int = 0) -> SimResult:
        rng = np.random.default_rng(seed)
        n = int(duration / dt)
        t = np.arange(n) * dt
        # State: y0..y5 (position + velocity for 3 sub-populations)
        y = np.zeros((6, n))
        y[0, 0] = 0.0

        C1, C2, C3, C4 = self.C, 0.8*self.C, 0.25*self.C, 0.25*self.C

        for i in range(1, n):
            p = rng.normal(p_mean, p_std)
            y[0, i] = y[0, i-1] + dt * y[3, i-1]
            y[1, i] = y[1, i-1] + dt * y[4, i-1]
            y[2, i] = y[2, i-1] + dt * y[5, i-1]
            y[3, i] = y[3, i-1] + dt * (self.A * self.a * self._sigm(y[1, i-1] - y[2, i-1])
                                         - 2 * self.a * y[3, i-1] - self.a**2 * y[0, i-1])
            y[4, i] = y[4, i-1] + dt * (self.A * self.a * (p + C2 * self._sigm(C1 * y[0, i-1]))
                                         - 2 * self.a * y[4, i-1] - self.a**2 * y[1, i-1])
            y[5, i] = y[5, i-1] + dt * (self.B * self.b * C4 * self._sigm(C3 * y[0, i-1])
                                         - 2 * self.b * y[5, i-1] - self.b**2 * y[2, i-1])

        V_eeg = y[1] - y[2]
        return SimResult(t=t, V=V_eeg, spike_times=np.array([]),
                         state_vars={"y": y})

    def step(self, inputs: float, state: Dict) -> Dict:
        return state
