"""FastAPI REST API for the Neurodynamics platform."""
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Neurodynamics API", version="1.0.0")


class SimulateRequest(BaseModel):
    model: str = "lif"
    duration: float = 500.0
    current: float = 5.0
    dt: float = 0.1
    params: Dict[str, Any] = {}


class SimulateResponse(BaseModel):
    model: str
    duration: float
    n_spikes: int
    spike_times: List[float]
    final_voltage: float


class AnalyzeRequest(BaseModel):
    data: List[float]
    analysis: str = "entropy"
    params: Dict[str, Any] = {}


class AnalyzeResponse(BaseModel):
    analysis: str
    result: Any


@app.get("/health")
def health():
    return {"status": "ok", "service": "neurodynamics"}


@app.get("/models")
def list_models():
    return {
        "models": [
            "lif", "adex", "hodgkin_huxley", "izhikevich",
            "fitzhugh_nagumo", "wilson_cowan", "neural_mass",
        ]
    }


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    import numpy as np
    from ..neural_dynamics.neuron_models import (
        LeakyIntegrateAndFire, AdaptiveExponential, HodgkinHuxley,
        Izhikevich, FitzHughNagumo, WilsonCowan,
    )
    model_map = {
        "lif": LeakyIntegrateAndFire,
        "adex": AdaptiveExponential,
        "hodgkin_huxley": HodgkinHuxley,
        "izhikevich": Izhikevich,
        "fitzhugh_nagumo": FitzHughNagumo,
        "wilson_cowan": WilsonCowan,
    }
    if req.model not in model_map:
        raise HTTPException(400, f"Unknown model: {req.model}")
    model = model_map[req.model](**req.params)
    result = model.simulate(req.duration, req.current, req.dt)
    return SimulateResponse(
        model=req.model,
        duration=req.duration,
        n_spikes=len(result.spike_times),
        spike_times=result.spike_times.tolist()[:50],
        final_voltage=float(result.V[-1]),
    )


@app.get("/experiments")
def list_experiments():
    return {
        "experiments": [
            "n_back", "maze", "delayed_match_to_sample", "stroop",
            "go_nogo", "iowa_casino", "working_memory_span",
        ]
    }


@app.post("/experiment")
def run_experiment(task_name: str = "n_back", n_steps: int = 20):
    import numpy as np
    from ..experiments.tasks import NBackTask, MazeTask, GoNoGoTask
    task_map = {
        "n_back": NBackTask,
        "maze": MazeTask,
        "go_nogo": GoNoGoTask,
    }
    if task_name not in task_map:
        raise HTTPException(400, f"Unknown task: {task_name}")
    task = task_map[task_name]()
    obs = task.reset()
    rewards = []
    for _ in range(n_steps):
        action = int(np.random.randint(0, 2))
        result = task.step(action)
        reward = result[1]
        done = result[2]
        rewards.append(reward)
        if done:
            break
    return {"task": task_name, "n_steps": len(rewards),
            "mean_reward": float(np.mean(rewards))}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    import numpy as np
    from ..dynamics.information import shannon_entropy, mutual_information
    from ..dynamics.chaos import sample_entropy, lyapunov_spectrum
    data = np.array(req.data)
    if req.analysis == "entropy":
        hist, _ = np.histogram(data, bins=20)
        p = hist / hist.sum()
        result = shannon_entropy(p)
    elif req.analysis == "sample_entropy":
        result = sample_entropy(data, **req.params)
    elif req.analysis == "lyapunov":
        traj = data.reshape(-1, 1)
        result = float(lyapunov_spectrum(traj)[0]) if len(traj) > 10 else 0.0
    else:
        raise HTTPException(400, f"Unknown analysis: {req.analysis}")
    return AnalyzeResponse(analysis=req.analysis, result=float(result))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
