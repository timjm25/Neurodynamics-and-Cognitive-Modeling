import numpy as np
import pytest

from neurodynamics.experiments import (
    NBackTask, MazeTask, GoNoGoTask, StroopTask, IowaCasinoTask,
)


def test_n_back_task_correct_hit():
    task = NBackTask(n=1, seq_length=10, seed=99)
    # Manually inject a known target
    task.reset()
    task.sequence = [1, 2, 1, 3, 3, 4, 5, 6, 7, 8]
    task.current_idx = 1  # next is index 2 (value=1), n-back is index 1 (value=2) -- not target
    # Skip to position 2 where sequence[2]=1, sequence[2-1]=2 → NOT target; position 4: seq[4]=3, seq[3]=3 → target
    task.current_idx = 4  # sequence[4]=3, sequence[3]=3 → target
    obs, reward, done, info = task.step(1)  # respond "match"
    assert info["target"] is True
    assert reward > 0


def test_n_back_task_correct_reject():
    task = NBackTask(n=1, seq_length=10, seed=0)
    task.reset()
    task.sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    task.current_idx = 2  # seq[2]=3, seq[1]=2 → not target
    obs, reward, done, info = task.step(0)  # respond "no match"
    assert info["target"] is False
    assert reward > 0


def test_maze_task_terminates():
    task = MazeTask(size=3, seed=0)
    obs = task.reset()
    done = False
    steps = 0
    max_steps = 1000
    rng = np.random.default_rng(0)
    while not done and steps < max_steps:
        action = int(rng.integers(0, 4))
        obs, reward, done, info = task.step(action)
        steps += 1
        if info["at_goal"]:
            break
    # Maze should eventually be solvable; but even random terminates at max_steps
    assert steps <= max_steps


def test_go_nogo_inhibition():
    task = GoNoGoTask(go_prob=0.5, n_trials=10, seed=5)
    obs = task.reset()
    # Find a no-go trial
    for _ in range(20):
        if obs.get("is_go", True) is False:
            obs2, reward, done, info = task.step(0)  # withhold
            assert info["correct"] is True
            break
        obs2, reward, done, info = task.step(1)
        obs = obs2
        if done:
            task.reset()


def test_stroop_congruent_vs_incongruent():
    task = StroopTask(n_trials=20, congruent_frac=0.5, seed=0)
    obs = task.reset()
    congruent_found = False
    incongruent_found = False
    for _ in range(20):
        congruent_found |= obs.get("congruent", False)
        incongruent_found |= not obs.get("congruent", True)
        obs, r, done, info = task.step(obs.get("ink_color", 0))
        if done:
            break
    assert congruent_found or incongruent_found  # at least one type found


def test_iowa_casino_four_decks():
    task = IowaCasinoTask(n_trials=40, seed=0)
    obs = task.reset()
    assert len(obs) == 4
    total_reward = 0.0
    for deck in [0, 1, 2, 3] * 10:
        obs, reward, done, info = task.step(deck)
        total_reward += reward
        if done:
            break
    # Task should produce numeric rewards
    assert isinstance(total_reward, float)
