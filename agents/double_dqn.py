from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from agents.dqn import DQNAgent

from utils.q_network import QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent to be trained.

    Parameters
    ----------
    env: Environment to train the agent on (optimized for MinAtar/Breakout-v1)
    gamma: Discount factor for future rewards
    lr: Learning rate
    batch_size: Number of samples to use for each learning step
    epsilon_min: Minimum value of epsilon for the epsilon-greedy policy
    epsilon_decay: Decay rate of epsilon per timestep
    memory_size: Number of samples to store in the replay buffer
    update_rate: Number of timesteps between each learning step
    target_update_rate: Number of timesteps between each update of the target network
    logging_directory: Directory to save the tensorboard logs to
    agent_name: Name of the agent
    log_images: Whether to log images of the last episode to tensorboard (won't work with another environment than MinAtar/Breakout-v1)
    checkpoint_directory: Directory to save the agent when it reaches a new best reward
    """
    def __init__(
            self,
            env: gym.Env,
            gamma: float = 0.99,
            lr: float = 0.001,
            batch_size: int = 32,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 0.999,
            memory_size: int = 100_000,
            update_rate:int = 4,
            target_update_rate=1000,
            logging_directory: str = './logs',
            agent_name: str = 'DDQN',
            log_images: bool = True,
            checkpoint_directory: str | None ='./networks_weights',

    ) -> DoubleDQNAgent:
        super().__init__(
            env=env,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            memory_size=memory_size,
            update_rate=update_rate,
            logging_directory=logging_directory,
            agent_name=agent_name,
            log_images=log_images,
            checkpoint_directory=checkpoint_directory
        )
        self.target_update_rate = target_update_rate
        self.target_network = QNetwork(self.state_shape, self.action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # initialise target network
        self.target_network.eval()  # set target network to eval mode

    def learn(self):
        """Update the Q-network.
        
        Returns
        -------
        loss: Loss of the network
        """
        experiences_batch = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences_batch
        # Use the online network to choose actions
        action_values = self.q_network(next_states).detach()
        action_indices = torch.argmax(action_values, dim=1, keepdim=True)
        # Use the target network to get corresponding action values
        action_values_target = self.target_network(next_states).detach()
        max_action_values_target = action_values_target.gather(1, action_indices)

        # If done just use reward, else update Q_target with discounted action values
        q_target = rewards + (self.gamma * max_action_values_target * (1 - dones))
        q_expected = self.q_network(states).gather(1, actions)

        loss = F.mse_loss(q_target, q_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.timestep % self.target_update_rate == 0:  # update target network weights with the policy network's ones
            self.target_network.load_state_dict(self.q_network.state_dict())
        return loss.item()

    @staticmethod
    def load(path: str) -> DoubleDQNAgent:
        """Load an agent from a file.

        Parameters
        ----------
        path: Path to load the agent from
        
        Returns
        -------
        agent: Loaded agent
        """
        agent: DoubleDQNAgent = super(DoubleDQNAgent, DoubleDQNAgent).load(path)
        agent.target_network.to(device)
        return agent
