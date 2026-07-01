from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import time


@dataclass
class Message:
    sender: str
    receiver: str  # "*" for broadcast
    message_type: str
    payload: Any
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._mailboxes: Dict[str, List[Message]] = {}
        self._history: List[Message] = []

    def register(self, agent_id: str) -> None:
        self._mailboxes[agent_id] = []
        self._handlers[agent_id] = []

    def subscribe(self, agent_id: str, handler: Callable) -> None:
        if agent_id not in self._handlers:
            self._handlers[agent_id] = []
        self._handlers[agent_id].append(handler)

    def send(self, message: Message) -> None:
        self._history.append(message)
        if message.receiver == "*":
            for aid, mailbox in self._mailboxes.items():
                if aid != message.sender:
                    mailbox.append(message)
                    for h in self._handlers.get(aid, []):
                        h(message)
        else:
            if message.receiver in self._mailboxes:
                self._mailboxes[message.receiver].append(message)
                for h in self._handlers.get(message.receiver, []):
                    h(message)

    def receive(self, agent_id: str) -> List[Message]:
        msgs = self._mailboxes.get(agent_id, []).copy()
        self._mailboxes[agent_id] = []
        return msgs

    def clear(self) -> None:
        for k in self._mailboxes:
            self._mailboxes[k] = []
        self._history.clear()


class BroadcastChannel:
    def __init__(self, bus: MessageBus, sender: str):
        self.bus = bus
        self.sender = sender

    def publish(self, msg_type: str, payload: Any) -> None:
        self.bus.send(Message(self.sender, "*", msg_type, payload))


class DirectChannel:
    def __init__(self, bus: MessageBus, sender: str, receiver: str):
        self.bus = bus
        self.sender = sender
        self.receiver = receiver

    def send(self, msg_type: str, payload: Any) -> None:
        self.bus.send(Message(self.sender, self.receiver, msg_type, payload))
