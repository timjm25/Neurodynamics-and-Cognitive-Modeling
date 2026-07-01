import threading
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Callable, Dict, List


class EventType(Enum):
    SIMULATION_START = auto()
    SIMULATION_STEP = auto()
    SIMULATION_END = auto()
    SPIKE = auto()
    LEARNING_UPDATE = auto()
    EXPERIMENT_RESET = auto()
    EXPERIMENT_STEP = auto()
    AGENT_MESSAGE = auto()
    HYPOTHESIS_GENERATED = auto()
    PLUGIN_LOADED = auto()
    ERROR = auto()


class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        with self._lock:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        with self._lock:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass

    def publish(self, event_type: EventType, data: Any = None) -> None:
        with self._lock:
            callbacks = list(self._subscribers[event_type])
        for cb in callbacks:
            cb(event_type, data)

    def clear(self, event_type: EventType = None) -> None:
        with self._lock:
            if event_type is None:
                self._subscribers.clear()
            else:
                self._subscribers[event_type].clear()


_global_bus = EventBus()


def get_global_bus() -> EventBus:
    return _global_bus
