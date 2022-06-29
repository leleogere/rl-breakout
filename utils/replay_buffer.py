from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Buffer to store experience tuples.
    
    Parameters
    ----------
    buffer_size: maximum size of internal memory
    batch_size: sample size from experience
    """
    def __init__(self, buffer_size: int, batch_size: int) -> ReplayBuffer:
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)

    def add(self, state: np.ndarray, action: int, reward: int, next_state: np.ndarray, done: bool) -> None:
        """Add experience"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a random sample from the memory.

        Returns
        -------
        states, actions, rewards, next_states, dones
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to torch tensors
        states = torch.from_numpy(np.stack([experience.state for experience in experiences if experience is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([experience.action for experience in experiences if experience is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([experience.next_state for experience in experiences if experience is not None])).float().to(device)
        # Convert done from boolean to int
        dones = torch.from_numpy(np.vstack([experience.done for experience in experiences if experience is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)

    def save(self, path: str) -> None:
        """Save the buffer to a file."""
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> ReplayBuffer:
        """Load a buffer from a file."""
        return torch.load(path)
