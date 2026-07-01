"""Disease models for clinical research extensions."""
from abc import ABC, abstractmethod
from typing import Any


class DiseaseModel(ABC):
    def __init__(self, severity: float = 0.5):
        self._severity = severity

    @property
    def severity(self) -> float:
        return self._severity

    @abstractmethod
    def apply(self, brain_network: Any) -> None:
        pass

    def description(self) -> str:
        return f"{self.__class__.__name__} (severity={self._severity:.2f})"


class AlzheimersDisease(DiseaseModel):
    def __init__(self, stage: float = 0.3):
        super().__init__(stage)
        self.tau_accumulation: float = stage
        self.amyloid_burden: float = stage * 0.8

    def apply(self, brain_network: Any) -> None:
        # Reduce hippocampal connectivity and plasticity
        hippo = brain_network.regions.get("Hippocampus")
        if hippo:
            hippo.connectivity_scale = max(0.1, 1.0 - self._severity)
        pfc = brain_network.regions.get("PFC")
        if pfc:
            pfc.wm.capacity = max(2, int(pfc.wm.capacity * (1 - self._severity * 0.5)))
            pfc.wm.decay_rate *= (1 + self._severity)


class ParkinsonsDisease(DiseaseModel):
    def __init__(self, severity: float = 0.4):
        super().__init__(severity)
        self.dopamine_deficit: float = severity
        self.tremor_frequency: float = 4.0 + 2.0 * severity  # 4-6 Hz tremor

    def apply(self, brain_network: Any) -> None:
        bg = brain_network.regions.get("BasalGanglia")
        if bg:
            bg.dopamine = max(0.1, 1.0 - self._severity)
        motor = brain_network.regions.get("MotorCortex")
        if motor:
            motor.gain = max(0.2, 1.0 - self._severity * 0.5)


class Schizophrenia(DiseaseModel):
    def __init__(self, subtype: str = "paranoid", severity: float = 0.4):
        super().__init__(severity)
        self.subtype = subtype
        self.hypofrontality: float = severity

    def apply(self, brain_network: Any) -> None:
        pfc = brain_network.regions.get("PFC")
        if pfc:
            pfc.control_gain = max(0.1, 1.0 - self._severity)
        hippo = brain_network.regions.get("Hippocampus")
        if hippo:
            hippo.connectivity_scale = max(0.3, 1.0 - self._severity * 0.3)


class Epilepsy(DiseaseModel):
    def __init__(self, seizure_threshold: float = 0.3):
        super().__init__(1.0 - seizure_threshold)
        self.seizure_threshold: float = seizure_threshold
        self.inhibitory_deficit: float = 1.0 - seizure_threshold

    def apply(self, brain_network: Any) -> None:
        # Lower inhibitory tone — simplified as raising dopamine/excitability
        bg = brain_network.regions.get("BasalGanglia")
        if bg:
            bg.dopamine = min(2.0, bg.dopamine * (1 + self.inhibitory_deficit))

    @property
    def severity(self) -> float:
        return self.inhibitory_deficit


class ADHD(DiseaseModel):
    def __init__(self, subtype: str = "combined", severity: float = 0.4):
        super().__init__(severity)
        self.subtype = subtype
        self.impulsivity: float = severity

    def apply(self, brain_network: Any) -> None:
        pfc = brain_network.regions.get("PFC")
        if pfc:
            pfc.control_gain = max(0.2, 1.0 - self._severity * 0.4)
        bg = brain_network.regions.get("BasalGanglia")
        if bg:
            bg.impulsivity = self.impulsivity


class Depression(DiseaseModel):
    def __init__(self, severity: float = 0.4):
        super().__init__(severity)
        self.anhedonia: float = severity
        self.reward_bias: float = -severity

    def apply(self, brain_network: Any) -> None:
        amyg = brain_network.regions.get("Amygdala")
        if amyg:
            amyg.fear_threshold = max(0.1, amyg.fear_threshold * (1 - self._severity * 0.3))
        bg = brain_network.regions.get("BasalGanglia")
        if bg:
            bg.dopamine = max(0.2, 1.0 - self._severity * 0.5)
