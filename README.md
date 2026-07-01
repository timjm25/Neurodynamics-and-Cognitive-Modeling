# Neurodynamics & Cognitive Modeling Research Platform

A modular, production-quality Python research platform for computational neuroscience, cognitive modeling, and AI-driven hypothesis discovery. Combine classical neuron models with modern machine learning, brain network analysis, and autonomous experiment design.

---

## Mission

Model, simulate, analyze, and discover principles governing neural computation across multiple scales — from single neurons to whole-brain networks to cognitive architectures. The platform bridges computational neuroscience, dynamical systems theory, and machine learning in a unified, extensible framework.

---

## Architecture

```
neurodynamics/
├── core/               — Config, event bus, plugin registry, ABCs
├── neural_dynamics/    — Neuron models, spiking networks, reservoir computing
├── connectivity/       — Graph construction, analysis, connectomes
├── cognitive/          — WM, attention, decision-making, predictive coding
├── dynamics/           — Chaos, bifurcation, information theory, topology
├── learning/           — Hebbian/STDP/BCM rules, Hopfield, RL agents
├── agents/             — Brain region agents, message bus, cognitive twins
├── experiments/        — Cognitive tasks (N-back, maze, Stroop, etc.)
├── data/               — Loaders (BIDS, NWB, CSV), synthetic generators
├── visualization/      — Raster plots, phase portraits, dashboards
├── hypothesis/         — Emergent behavior detection, symbolic regression
├── clinical/           — Disease models (Alzheimer's, Parkinson's, etc.)
├── api/                — FastAPI REST endpoints + MCP interface
├── sdk/                — Plugin SDK for extending the platform
└── cli/                — Command-line interface
```

---

## Quick Start

```bash
git clone https://github.com/timjm25/Neurodynamics-and-Cognitive-Modeling
cd Neurodynamics-and-Cognitive-Modeling

# Install dependencies
python3 -m pip install -r requirements.txt

# Run example: LIF simulation
python3 examples/01_lif_simulation.py

# Run tests
python3 -m pytest tests/ -v
# 66 passed in ~3s

# Start REST API
python3 -m neurodynamics.api.rest
# → http://localhost:8000
```

---

## Neural Dynamics Module

### Implemented Neuron Models

| Model | Parameters | Features |
|-------|-----------|---------|
| `LeakyIntegrateAndFire` | τ_m, V_rest, V_thresh, V_reset, R_m | Spike reset, current injection |
| `AdaptiveExponential` (AdEx) | C, g_L, E_L, V_T, Δ_T, τ_w, a, b | Spike frequency adaptation |
| `HodgkinHuxley` | g_Na, g_K, g_L, E_Na, E_K, E_L, C_m | Full conductance model, m/h/n gates |
| `Izhikevich` | a, b, c, d | Efficient spiking, multiple patterns |
| `FitzHughNagumo` | a, b, τ, I_ext | 2D excitable system |
| `WilsonCowan` | τ_e, τ_i, w_ee, w_ei, w_ie, w_ii | E/I population dynamics |
| `NeuralMassModel` | A, B, a, b, C | Jansen-Rit cortical column |

```python
from neurodynamics.neural_dynamics import LeakyIntegrateAndFire, HodgkinHuxley

# LIF simulation
lif = LeakyIntegrateAndFire(tau_m=20.0, v_thresh=-55.0)
result = lif.simulate(duration=500.0, current=3.0, dt=0.1)
print(f"Spikes: {len(result.spike_times)}")

# Hodgkin-Huxley action potential
hh = HodgkinHuxley()
result = hh.simulate(duration=50.0, current=20.0)
```

### Reservoir Computing

```python
from neurodynamics.neural_dynamics import EchoStateNetwork
import numpy as np

esn = EchoStateNetwork(input_dim=1, reservoir_dim=200, spectral_radius=0.9)
inp = np.sin(np.linspace(0, 6*np.pi, 400))
target = np.cos(np.linspace(0, 6*np.pi, 400))
states = esn.run(inp, washout=100)
esn.train_readout(states, target[100:])
pred = esn.predict(inp, washout=100)
```

---

## Brain Connectivity

```python
from neurodynamics.connectivity import (
    build_small_world, build_scale_free, build_modular,
    functional_connectivity, community_detection_louvain,
    spectral_clustering, clustering_coefficient,
)

# Build networks
W_sw = build_small_world(n=100, k=6, p=0.1)   # Watts-Strogatz
W_sf = build_scale_free(n=100, m=2)             # Barabasi-Albert
W_mod = build_modular(n_modules=4, n_per_module=25, p_intra=0.4, p_inter=0.02)

# Analyze
fc = functional_connectivity(time_series)        # Pearson correlation
communities = community_detection_louvain(W_mod)  # Louvain
cc = clustering_coefficient(W_sw)               # Per-node CC
```

---

## Cognitive Modeling

### Working Memory

```python
from neurodynamics.cognitive import WorkingMemoryModel, ContinuousAttractorWM

wm = WorkingMemoryModel(capacity=7, decay_rate=0.05)
wm.encode("item_A")
wm.forget(dt=10.0)

ring_wm = ContinuousAttractorWM(n=100)
ring_wm.set_memory(position=1.5)
pos = ring_wm.read_out()
```

### Decision Making

```python
from neurodynamics.cognitive import DriftDiffusionModel, BayesianDecisionMaker, RaceModel

ddm = DriftDiffusionModel(drift=1.5, noise=1.0, threshold=1.0)
choice, rt = ddm.decide(stimulus_strength=2.0)

bdm = BayesianDecisionMaker(n_hypotheses=3)
bdm.update([0.1, 0.8, 0.1])
decision = bdm.decide()  # → 1

race = RaceModel(n_alternatives=4)
winner, rt = race.compete(np.array([0.5, 2.0, 0.3, 0.1]))
```

### Predictive Coding / Active Inference

```python
from neurodynamics.cognitive import PredictiveCodingLayer, ActiveInferenceAgent

pc = PredictiveCodingLayer(dim=10, lr=0.1)
pe = pc.forward(sensory_input)   # returns prediction errors

agent = ActiveInferenceAgent(n_states=4, n_actions=2)
agent.infer_state(observation)
action = agent.select_action(preferences)
```

---

## Dynamical Systems Laboratory

```python
from neurodynamics.dynamics import (
    simulate_lorenz, lyapunov_spectrum, is_chaotic,
    correlation_dimension, sample_entropy,
    shannon_entropy, mutual_information, transfer_entropy,
    BifurcationScanner,
)

# Chaos analysis
traj = simulate_lorenz(duration=100.0)
les = lyapunov_spectrum(traj)
print(f"Chaotic: {is_chaotic(les)}")  # True
D2 = correlation_dimension(traj)

# Information theory
H = shannon_entropy(prob_dist)
MI = mutual_information(x, y, bins=20)
TE = transfer_entropy(source, target, lag=1)

# Bifurcation analysis
scanner = BifurcationScanner()
diagram = scanner.scan(model_fn, "r", np.linspace(2.5, 4.0, 50),
                        initial_state=np.array([0.5]))
```

---

## Learning Rules

```python
from neurodynamics.learning import HebbianRule, STDPRule, OjaRule, HopfieldNetwork

# Hebbian learning
hebb = HebbianRule()
W_new = hebb.update(W, pre, post, lr=0.01)

# STDP
stdp = STDPRule()
W_new = stdp.update(W, pre_spikes, post_spikes, A_plus=0.01, A_minus=0.012)

# Hopfield associative memory
net = HopfieldNetwork(n=100)
net.store(patterns)               # store up to ~14 patterns
recalled = net.recall(corrupted)  # pattern completion
```

---

## Multi-Agent Brain Simulation

```python
from neurodynamics.agents import BrainNetwork, CognitiveTwinSimulator

# Full brain network
brain = BrainNetwork()
brain.send("external", "PFC", "encode", "working_memory_item")
brain.send("external", "Hippocampus", "store", pattern.tolist())
state = brain.step()

# Cognitive digital twin
sim = CognitiveTwinSimulator()
twin = sim.create({"memory_capacity": 7, "age_years": 30, "plasticity": 1.0})
result = sim.run_task(twin, n_back_task)
sim.simulate_aging(twin, years=30)
sim.apply_lesion(twin, "Hippocampus", severity=0.5)
```

---

## Experiment Tasks

```python
from neurodynamics.experiments import NBackTask, MazeTask, StroopTask, GoNoGoTask, IowaCasinoTask

# N-Back working memory
task = NBackTask(n=2, seq_length=50)
obs = task.reset()
obs, reward, done, info = task.step(action=1)  # 1=match, 0=no-match

# Maze navigation
maze = MazeTask(size=10)
obs = maze.reset()
obs, reward, done, info = maze.step(action=3)  # 0=up,1=down,2=left,3=right

# Decision tasks
stroop = StroopTask(n_trials=40)
igt = IowaCasinoTask(n_trials=100)
```

---

## Clinical Research Extensions

```python
from neurodynamics.clinical import AlzheimersDisease, ParkinsonsDisease, Schizophrenia, Epilepsy, ADHD
from neurodynamics.agents import BrainNetwork

net = BrainNetwork()

# Apply disease model
ad = AlzheimersDisease(stage=0.6)
ad.apply(net)  # Reduces hippocampal connectivity, increases decay

pd = ParkinsonsDisease(severity=0.5)
pd.apply(net)  # Reduces dopamine in basal ganglia

# Compare healthy vs diseased performance
```

---

## Hypothesis Discovery

```python
from neurodynamics.hypothesis import HypothesisDiscoveryEngine

engine = HypothesisDiscoveryEngine()
behaviors = engine.detect_emergent_behaviors(time_series)
attractors = engine.find_attractors([trajectory])
eq = engine.symbolic_regression(X, y)             # "y = 1.000 + 2.000*x^1"
hypotheses = engine.generate_hypothesis(data)     # ranked list
sensitivity = engine.scan_parameter_space(model, {"gain": np.linspace(0, 2, 20)}, metric_fn)
```

---

## REST API

```bash
# Start server
python3 neurodynamics/api/rest.py

# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Simulate LIF neuron
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"model":"lif","duration":500,"current":3.0}'

# Analyze entropy of time series
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"data":[0.1,0.5,0.9,0.3,...],"analysis":"entropy"}'
```

---

## CLI Reference

```
python3 -m neurodynamics.cli.main simulate --model lif --duration 500 --current 3.0
python3 -m neurodynamics.cli.main analyze --data "[1,2,3,...]" --analysis entropy
python3 -m neurodynamics.cli.main serve --host 0.0.0.0 --port 8000
```

---

## Plugin SDK

```python
from neurodynamics.sdk import NeuronModelPlugin, validate_plugin

class MyNeuronPlugin(NeuronModelPlugin):
    name = "my_neuron"
    description = "Custom neuron model"

    def step(self, inputs, state):
        V = state.get("V", -70.0)
        return {"V": V + inputs * 0.1, "spiked": V > -55.0}

    def simulate(self, duration, inputs):
        # ... implementation
        return result

valid, errors = validate_plugin(MyNeuronPlugin())
```

---

## Docker Deployment

```bash
docker build -t neurodynamics -f docker/Dockerfile .
docker run -p 8000:8000 neurodynamics

# Or with compose
docker-compose -f docker/docker-compose.yml up
```

---

## Testing

```bash
python3 -m pytest tests/ -v
# 66 tests across neural dynamics, connectivity, cognitive, dynamics,
# learning, agents, experiments, hypothesis, clinical, API
```

Test coverage spans:
- All neuron model ODEs (LIF, AdEx, HH, Izhikevich, FHN, Wilson-Cowan)
- Echo State Network training
- STDP potentiation and depression
- Graph construction and analysis
- Cognitive task correctness
- Lorenz chaos detection
- Disease model parameter effects
- REST API endpoints

---

## Roadmap

- [ ] GPU acceleration (JAX/CUDA backends)
- [ ] Whole-brain connectome integration (HCP, Allen Atlas)
- [ ] Real EEG/fMRI data loaders (MNE-Python, NiBabel)
- [ ] Visual workflow editor (node-based)
- [ ] UMAP/t-SNE latent manifold explorer
- [ ] LLM-powered AI research assistant
- [ ] Kubernetes deployment chart
- [ ] Distributed simulation (Ray, Dask)
- [ ] VR-ready 3D brain viewer
- [ ] Benchmark suite against NEST, Brian2, NetPyNE

---

## References

1. Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *J. Physiol.* 117, 500–544.
2. Izhikevich, E.M. (2003). Simple model of spiking neurons. *IEEE Trans. Neural Netw.* 14(6), 1569–1572.
3. Jaeger, H. & Haas, H. (2004). Harnessing nonlinearity. *Science* 304, 78–80.
4. Watts, D.J. & Strogatz, S.H. (1998). Collective dynamics of small-world networks. *Nature* 393, 440–442.
5. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nat. Rev. Neurosci.* 11, 127–138.
6. Jansen, B.H. & Rit, V.G. (1995). Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. *Biol. Cybern.* 73, 357–366.
7. Buzsáki, G. & Draguhn, A. (2004). Neuronal oscillations in cortical networks. *Science* 304, 1926–1929.
8. Maass, W., Natschläger, T. & Markram, H. (2002). Real-time computing without stable states. *Neural Comput.* 14(11), 2531–2560.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for frontier neuroscience and AI research. Designed to be extended — every module is replaceable via the Plugin SDK.*
