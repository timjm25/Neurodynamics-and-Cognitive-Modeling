"""Reinforcement learning agents for cognitive tasks."""
from typing import Any, Dict, Optional

import numpy as np


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1,
                 gamma: float = 0.9, epsilon: float = 0.1, seed: int = 0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int) -> None:
        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

    def reset(self) -> None:
        self.Q = np.zeros((self.n_states, self.n_actions))


class ActorCriticAgent:
    def __init__(self, n_states: int, n_actions: int,
                 lr_actor: float = 0.01, lr_critic: float = 0.1,
                 gamma: float = 0.95, seed: int = 0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.V = np.zeros(n_states)
        self.policy = np.ones((n_states, n_actions)) / n_actions

    def select_action(self, state: int) -> int:
        probs = self.policy[state]
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        return int(self.rng.choice(self.n_actions, p=probs))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        td_error = (reward + (0 if done else self.gamma * self.V[next_state])
                    - self.V[state])
        self.V[state] += self.lr_critic * td_error
        self.policy[state, action] += self.lr_actor * td_error
        self.policy[state] = np.clip(self.policy[state], 0.01, None)
        self.policy[state] /= self.policy[state].sum()
