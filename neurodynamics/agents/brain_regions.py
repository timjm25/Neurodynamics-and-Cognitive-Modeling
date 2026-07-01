"""Modular brain region agents communicating via MessageBus."""
from typing import Any, Dict, List, Optional

import numpy as np

from .message_bus import Message, MessageBus


class BrainRegionAgent:
    def __init__(self, name: str, bus: MessageBus):
        self.name = name
        self.bus = bus
        self._inbox: List[Message] = []
        bus.register(name)
        bus.subscribe(name, self._on_message)

    def _on_message(self, msg: Message) -> None:
        self._inbox.append(msg)

    def receive(self) -> List[Message]:
        msgs = self._inbox.copy()
        self._inbox.clear()
        return msgs

    def process(self) -> Any:
        raise NotImplementedError

    def send(self, receiver: str, msg_type: str, payload: Any) -> None:
        self.bus.send(Message(self.name, receiver, msg_type, payload))

    def broadcast(self, msg_type: str, payload: Any) -> None:
        self.bus.send(Message(self.name, "*", msg_type, payload))


class PrefrontalCortex(BrainRegionAgent):
    def __init__(self, bus: MessageBus, wm_capacity: int = 7):
        super().__init__("PFC", bus)
        from ..cognitive.working_memory import WorkingMemoryModel
        from ..cognitive.decision import DriftDiffusionModel
        self.wm = WorkingMemoryModel(capacity=wm_capacity)
        self.ddm = DriftDiffusionModel()
        self.control_gain: float = 1.0

    def process(self) -> Dict:
        msgs = self.receive()
        for msg in msgs:
            if msg.message_type == "encode":
                self.wm.encode(msg.payload)
            elif msg.message_type == "forget":
                self.wm.forget(float(msg.payload))
        return {"wm_load": self.wm.load, "control_gain": self.control_gain}

    def decide(self, stimulus: float) -> tuple:
        return self.ddm.decide(stimulus * self.control_gain)

    def apply_lesion(self, severity: float) -> None:
        self.control_gain = max(0.0, 1.0 - severity)
        self.wm.capacity = max(1, int(self.wm.capacity * (1 - severity)))


class Hippocampus(BrainRegionAgent):
    def __init__(self, bus: MessageBus, capacity: int = 100):
        super().__init__("Hippocampus", bus)
        from ..learning.hebbian import HopfieldNetwork
        self.memory = HopfieldNetwork(capacity)
        self.patterns_stored: List[np.ndarray] = []
        self.connectivity_scale: float = 1.0

    def store_pattern(self, pattern: np.ndarray) -> None:
        self.patterns_stored.append(pattern)
        self.memory.store(self.patterns_stored)

    def recall(self, cue: np.ndarray) -> np.ndarray:
        return self.memory.recall(cue * self.connectivity_scale)

    def process(self) -> Dict:
        msgs = self.receive()
        for msg in msgs:
            if msg.message_type == "store":
                self.store_pattern(np.asarray(msg.payload))
            elif msg.message_type == "recall":
                recalled = self.recall(np.asarray(msg.payload))
                self.send(msg.sender, "recalled", recalled.tolist())
        return {"patterns_stored": len(self.patterns_stored)}

    def apply_lesion(self, severity: float) -> None:
        self.connectivity_scale = max(0.0, 1.0 - severity)


class BasalGanglia(BrainRegionAgent):
    def __init__(self, bus: MessageBus, n_actions: int = 4):
        super().__init__("BasalGanglia", bus)
        self.n_actions = n_actions
        self.dopamine: float = 1.0
        self.salience = np.zeros(n_actions)
        self.impulsivity: float = 0.1

    def select_action(self, inputs: np.ndarray) -> int:
        weighted = inputs * self.dopamine + self.impulsivity * np.random.randn(len(inputs))
        return int(np.argmax(weighted))

    def process(self) -> Dict:
        msgs = self.receive()
        for msg in msgs:
            if msg.message_type == "salience":
                self.salience = np.asarray(msg.payload)
        action = self.select_action(self.salience)
        self.broadcast("action_selected", action)
        return {"selected_action": action, "dopamine": self.dopamine}

    def apply_lesion(self, severity: float) -> None:
        self.dopamine = max(0.0, 1.0 - severity)


class Thalamus(BrainRegionAgent):
    def __init__(self, bus: MessageBus):
        super().__init__("Thalamus", bus)
        self.gate: float = 1.0

    def relay(self, signal: np.ndarray) -> np.ndarray:
        return signal * self.gate

    def process(self) -> Dict:
        msgs = self.receive()
        relayed = []
        for msg in msgs:
            if msg.message_type == "relay":
                out = self.relay(np.asarray(msg.payload))
                self.broadcast("relayed", out.tolist())
                relayed.append(out)
        return {"gate": self.gate}


class Amygdala(BrainRegionAgent):
    def __init__(self, bus: MessageBus):
        super().__init__("Amygdala", bus)
        self.fear_threshold: float = 0.5

    def assign_valence(self, stimulus: float) -> float:
        if stimulus > self.fear_threshold:
            return -1.0  # aversive
        return 1.0  # neutral/positive

    def process(self) -> Dict:
        msgs = self.receive()
        for msg in msgs:
            if msg.message_type == "stimulus":
                valence = self.assign_valence(float(msg.payload))
                self.broadcast("valence", valence)
        return {"fear_threshold": self.fear_threshold}


class Cerebellum(BrainRegionAgent):
    def __init__(self, bus: MessageBus, n_inputs: int = 10):
        super().__init__("Cerebellum", bus)
        self.n_inputs = n_inputs
        self.W = np.zeros((1, n_inputs))
        self.lr: float = 0.01

    def predict(self, context: np.ndarray) -> float:
        return float(self.W @ context[:self.n_inputs])

    def learn(self, context: np.ndarray, error: float) -> None:
        self.W -= self.lr * error * context[:self.n_inputs]

    def process(self) -> Dict:
        msgs = self.receive()
        for msg in msgs:
            if msg.message_type == "motor_context":
                pred = self.predict(np.asarray(msg.payload))
                self.broadcast("motor_prediction", pred)
        return {"weights_norm": float(np.linalg.norm(self.W))}


class VisualCortex(BrainRegionAgent):
    def __init__(self, bus: MessageBus, n_features: int = 16):
        super().__init__("VisualCortex", bus)
        self.n_features = n_features
        self.rng = np.random.default_rng(0)
        self.filters = self.rng.normal(0, 1, (n_features, n_features))

    def process_image(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=float).flatten()
        img = img[:self.n_features] if len(img) >= self.n_features else np.pad(img, (0, self.n_features - len(img)))
        return np.maximum(0, self.filters @ img)

    def process(self) -> Dict:
        msgs = self.receive()
        for msg in msgs:
            if msg.message_type == "visual_input":
                features = self.process_image(np.asarray(msg.payload))
                self.broadcast("visual_features", features.tolist())
        return {"n_features": self.n_features}


class MotorCortex(BrainRegionAgent):
    def __init__(self, bus: MessageBus, n_muscles: int = 4):
        super().__init__("MotorCortex", bus)
        self.n_muscles = n_muscles
        self.gain: float = 1.0

    def generate_command(self, goal: np.ndarray) -> np.ndarray:
        cmd = np.asarray(goal, dtype=float)[:self.n_muscles]
        if len(cmd) < self.n_muscles:
            cmd = np.pad(cmd, (0, self.n_muscles - len(cmd)))
        return cmd * self.gain

    def process(self) -> Dict:
        msgs = self.receive()
        for msg in msgs:
            if msg.message_type == "motor_goal":
                cmd = self.generate_command(np.asarray(msg.payload))
                self.broadcast("motor_command", cmd.tolist())
        return {"gain": self.gain}


class BrainNetwork:
    """Container for all brain region agents + message bus."""

    def __init__(self):
        self.bus = MessageBus()
        self.regions: Dict[str, BrainRegionAgent] = {}
        self._setup_default_regions()

    def _setup_default_regions(self) -> None:
        self.regions["PFC"] = PrefrontalCortex(self.bus)
        self.regions["Hippocampus"] = Hippocampus(self.bus)
        self.regions["BasalGanglia"] = BasalGanglia(self.bus)
        self.regions["Thalamus"] = Thalamus(self.bus)
        self.regions["Amygdala"] = Amygdala(self.bus)
        self.regions["Cerebellum"] = Cerebellum(self.bus)
        self.regions["VisualCortex"] = VisualCortex(self.bus)
        self.regions["MotorCortex"] = MotorCortex(self.bus)

    def step(self) -> Dict[str, Any]:
        return {name: region.process() for name, region in self.regions.items()}

    def send(self, sender: str, receiver: str, msg_type: str, payload: Any) -> None:
        self.bus.send(Message(sender, receiver, msg_type, payload))

    def apply_lesion(self, region_name: str, severity: float) -> None:
        region = self.regions.get(region_name)
        if hasattr(region, "apply_lesion"):
            region.apply_lesion(severity)
