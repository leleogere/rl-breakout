import torch
import numpy as np
import random
from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        Replay memory allow agent to record experiences and learn from them

        Parameters
        ---------
        buffer_size (int): maximum size of internal memory
        batch_size (int): sample size from experience
        """
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """
        Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors
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

    def __len__(self):
        return len(self.memory)

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)
