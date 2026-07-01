"""Neurodynamics & Cognitive Modeling Research Platform."""

__version__ = "1.0.0"
__author__ = "Neurodynamics Platform"

from .core import (
    SimulationConfig, PlatformConfig, EventBus, EventType, PluginRegistry,
    NeuronModel, NetworkModel, CognitiveModule,
)
from .neural_dynamics import (
    LeakyIntegrateAndFire, AdaptiveExponential, HodgkinHuxley,
    Izhikevich, FitzHughNagumo, WilsonCowan, NeuralMassModel,
    EchoStateNetwork, SpikeTrain, SpikingNetwork,
)
from .connectivity import (
    build_small_world, build_scale_free, build_modular,
    functional_connectivity, community_detection_louvain,
)
from .cognitive import (
    WorkingMemoryModel, DriftDiffusionModel, BayesianDecisionMaker,
    PredictiveCodingLayer, ActiveInferenceAgent,
)
from .dynamics import (
    lorenz_system, simulate_lorenz, lyapunov_spectrum, is_chaotic,
    shannon_entropy, mutual_information, transfer_entropy,
    BifurcationScanner,
)
from .learning import HebbianRule, STDPRule, HopfieldNetwork
from .agents import BrainNetwork, CognitiveTwin, CognitiveTwinSimulator
from .experiments import NBackTask, MazeTask, GoNoGoTask
from .clinical import (
    AlzheimersDisease, ParkinsonsDisease, Schizophrenia, Epilepsy, ADHD,
)
from .hypothesis import HypothesisDiscoveryEngine

__all__ = [
    "SimulationConfig", "PlatformConfig", "EventBus", "EventType", "PluginRegistry",
    "NeuronModel", "NetworkModel", "CognitiveModule",
    "LeakyIntegrateAndFire", "AdaptiveExponential", "HodgkinHuxley",
    "Izhikevich", "FitzHughNagumo", "WilsonCowan", "NeuralMassModel",
    "EchoStateNetwork", "SpikeTrain", "SpikingNetwork",
    "build_small_world", "build_scale_free", "build_modular",
    "functional_connectivity", "community_detection_louvain",
    "WorkingMemoryModel", "DriftDiffusionModel", "BayesianDecisionMaker",
    "PredictiveCodingLayer", "ActiveInferenceAgent",
    "lorenz_system", "simulate_lorenz", "lyapunov_spectrum", "is_chaotic",
    "shannon_entropy", "mutual_information", "transfer_entropy",
    "BifurcationScanner",
    "HebbianRule", "STDPRule", "HopfieldNetwork",
    "BrainNetwork", "CognitiveTwin", "CognitiveTwinSimulator",
    "NBackTask", "MazeTask", "GoNoGoTask",
    "AlzheimersDisease", "ParkinsonsDisease", "Schizophrenia", "Epilepsy", "ADHD",
    "HypothesisDiscoveryEngine",
]
