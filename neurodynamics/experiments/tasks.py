"""Cognitive and behavioral experiment tasks."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class NBackTask:
    def __init__(self, n: int = 2, seq_length: int = 20, n_stimuli: int = 9,
                 seed: int = 42):
        self.n = n
        self.seq_length = seq_length
        self.n_stimuli = n_stimuli
        self.rng = np.random.default_rng(seed)
        self.sequence: List[int] = []
        self.current_idx: int = 0

    def reset(self) -> Any:
        self.sequence = list(self.rng.integers(0, self.n_stimuli, self.seq_length))
        # Inject targets (~30%)
        for i in range(self.n, self.seq_length):
            if self.rng.random() < 0.3:
                self.sequence[i] = self.sequence[i - self.n]
        self.current_idx = self.n
        return self.sequence[self.current_idx - 1]

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        # action 1 = "match", 0 = "no match"
        is_target = (self.current_idx < self.seq_length and
                     self.sequence[self.current_idx] == self.sequence[self.current_idx - self.n])
        correct = (action == 1 and is_target) or (action == 0 and not is_target)
        reward = 1.0 if correct else -1.0
        self.current_idx += 1
        done = self.current_idx >= self.seq_length
        obs = self.sequence[self.current_idx - 1] if not done else 0
        return obs, reward, done, {"target": is_target}

    def is_target(self) -> bool:
        if self.current_idx < self.n or self.current_idx >= self.seq_length:
            return False
        return self.sequence[self.current_idx] == self.sequence[self.current_idx - self.n]


class MazeTask:
    def __init__(self, size: int = 5, seed: int = 42):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.agent_pos: List[int] = [0, 0]
        self.goal_pos: List[int] = [size - 1, size - 1]
        self._step_count: int = 0
        self._max_steps: int = size * size * 4

    def reset(self) -> np.ndarray:
        self.agent_pos = [0, 0]
        self._step_count = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = np.zeros((self.size, self.size))
        obs[self.agent_pos[0], self.agent_pos[1]] = 1.0
        obs[self.goal_pos[0], self.goal_pos[1]] = 2.0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # actions: 0=up, 1=down, 2=left, 3=right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = moves[action % 4]
        new_r = np.clip(self.agent_pos[0] + dr, 0, self.size - 1)
        new_c = np.clip(self.agent_pos[1] + dc, 0, self.size - 1)
        self.agent_pos = [new_r, new_c]
        self._step_count += 1
        at_goal = self.agent_pos == self.goal_pos
        done = at_goal or self._step_count >= self._max_steps
        reward = 10.0 if at_goal else -0.1
        return self._obs(), reward, done, {"at_goal": at_goal}


class DelayedMatchToSample:
    def __init__(self, delay: int = 5, n_stimuli: int = 8, seed: int = 0):
        self.delay = delay
        self.n_stimuli = n_stimuli
        self.rng = np.random.default_rng(seed)
        self._target: Optional[int] = None
        self._phase: str = "encode"
        self._delay_count: int = 0

    def reset(self) -> np.ndarray:
        self._target = int(self.rng.integers(0, self.n_stimuli))
        self._phase = "encode"
        self._delay_count = 0
        obs = np.zeros(self.n_stimuli)
        obs[self._target] = 1.0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._phase == "encode":
            self._phase = "delay"
            return np.zeros(self.n_stimuli), 0.0, False, {"phase": "delay"}
        elif self._phase == "delay":
            self._delay_count += 1
            if self._delay_count >= self.delay:
                self._phase = "match"
            return np.zeros(self.n_stimuli), 0.0, False, {"phase": "delay"}
        else:
            correct = int(action) == self._target
            reward = 1.0 if correct else -1.0
            return np.zeros(self.n_stimuli), reward, True, {"correct": correct}


class StroopTask:
    COLORS = ["red", "green", "blue", "yellow"]
    COLOR_IDX = {c: i for i, c in enumerate(COLORS)}

    def __init__(self, n_trials: int = 20, congruent_frac: float = 0.5,
                 seed: int = 0):
        self.n_trials = n_trials
        self.congruent_frac = congruent_frac
        self.rng = np.random.default_rng(seed)
        self._trial = 0
        self._congruent: bool = True
        self._ink_color: int = 0
        self._word: int = 0

    def reset(self) -> Dict:
        self._trial = 0
        return self._make_trial()

    def _make_trial(self) -> Dict:
        self._congruent = self.rng.random() < self.congruent_frac
        self._ink_color = int(self.rng.integers(0, len(self.COLORS)))
        if self._congruent:
            self._word = self._ink_color
        else:
            others = [i for i in range(len(self.COLORS)) if i != self._ink_color]
            self._word = int(self.rng.choice(others))
        return {"ink_color": self._ink_color, "word": self._word,
                "congruent": self._congruent}

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        correct = int(action) == self._ink_color
        reward = 1.0 if correct else -1.0
        self._trial += 1
        done = self._trial >= self.n_trials
        obs = self._make_trial() if not done else {}
        return obs, reward, done, {"congruent": self._congruent, "correct": correct}


class GoNoGoTask:
    def __init__(self, go_prob: float = 0.7, n_trials: int = 30, seed: int = 0):
        self.go_prob = go_prob
        self.n_trials = n_trials
        self.rng = np.random.default_rng(seed)
        self._trial = 0
        self._is_go: bool = True

    def reset(self) -> Dict:
        self._trial = 0
        self._is_go = self.rng.random() < self.go_prob
        return {"stimulus": "go" if self._is_go else "nogo", "is_go": self._is_go}

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        # action 1 = respond (go), 0 = withhold (no-go)
        if self._is_go:
            correct = action == 1
        else:
            correct = action == 0  # correct inhibition
        reward = 1.0 if correct else -1.0
        self._trial += 1
        done = self._trial >= self.n_trials
        self._is_go = self.rng.random() < self.go_prob
        obs = {"stimulus": "go" if self._is_go else "nogo", "is_go": self._is_go}
        return obs, reward, done, {"correct": correct}


class IowaCasinoTask:
    """Simplified Iowa Gambling Task — 4 decks."""
    # (mean_reward, std_reward, loss_prob, loss_magnitude)
    DECKS = [
        (1.0, 0.5, 0.5, 2.5),   # A: high reward, high loss → bad long-term
        (0.5, 0.25, 0.1, 1.25), # B: medium reward, low loss → bad long-term
        (0.5, 0.2, 0.5, 0.5),   # C: medium reward, small loss → good long-term
        (0.25, 0.1, 0.1, 0.25), # D: low reward, low loss → good long-term
    ]

    def __init__(self, n_trials: int = 100, seed: int = 0):
        self.n_trials = n_trials
        self.rng = np.random.default_rng(seed)
        self._trial = 0
        self._deck_counts = np.zeros(4)

    def reset(self) -> np.ndarray:
        self._trial = 0
        self._deck_counts = np.zeros(4)
        return np.zeros(4)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        deck_idx = int(action) % 4
        mu, std, loss_p, loss_mag = self.DECKS[deck_idx]
        reward = float(self.rng.normal(mu, std))
        if self.rng.random() < loss_p:
            reward -= loss_mag
        self._deck_counts[deck_idx] += 1
        self._trial += 1
        done = self._trial >= self.n_trials
        obs = self._deck_counts.copy()
        return obs, reward, done, {"deck": deck_idx}


class WorkingMemorySpan:
    def __init__(self, max_span: int = 9, seed: int = 0):
        self.max_span = max_span
        self.rng = np.random.default_rng(seed)
        self._sequence: List[int] = []
        self._recall_idx: int = 0
        self._phase: str = "encode"
        self._span: int = 3

    def reset(self) -> Dict:
        self._span = 3
        self._sequence = list(self.rng.integers(1, 10, self._span))
        self._recall_idx = 0
        self._phase = "encode"
        return {"sequence": self._sequence.copy(), "phase": "encode"}

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        if self._phase == "encode":
            self._phase = "recall"
            return {"phase": "recall", "position": 0}, 0.0, False, {}
        correct = int(action) == self._sequence[self._recall_idx]
        reward = 1.0 if correct else -1.0
        self._recall_idx += 1
        done = self._recall_idx >= len(self._sequence)
        if done and correct and self._span < self.max_span:
            self._span += 1
        obs = {"position": self._recall_idx, "phase": "recall"}
        return obs, reward, done, {"correct": correct, "span": self._span}
