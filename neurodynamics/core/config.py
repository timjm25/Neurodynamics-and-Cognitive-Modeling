from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimulationConfig:
    dt: float = 0.1          # ms
    duration: float = 1000.0  # ms
    seed: Optional[int] = 42
    device: str = "cpu"
    precision: str = "float64"


@dataclass
class PlatformConfig:
    data_dir: str = "./data"
    output_dir: str = "./output"
    plugin_dirs: List[str] = field(default_factory=list)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
