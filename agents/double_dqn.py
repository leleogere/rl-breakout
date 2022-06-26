import torch
import torch.nn.functional as F
import os

from agents.dqn import DQNAgent
from utils.q_network import QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleDQNAgent(DQNAgent):
    def __init__(
            self,
            env,
            gamma=0.99,
            lr=0.001,
            batch_size=32,
            epsilon_min=0.01,
            epsilon_decay=0.999,
            memory_size=100_000,
            update_rate=4,
            target_update_rate=1000,
            logging_directory='./logs',
            agent_name='DDQN',
            log_images=True,
            checkpoint_directory='./networks_weights',

    ):
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
        experiences_batch = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences_batch
        # Use the online network to choose action
        action_values = self.q_network(next_states).detach()
        action_indices = torch.argmax(action_values, dim=1, keepdim=True)
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
    def load(path) -> 'DoubleDQNAgent':
        return super(DoubleDQNAgent, DoubleDQNAgent).load(path)
