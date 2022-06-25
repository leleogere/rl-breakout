import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import os
from tqdm import tqdm

from utils.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
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
            logging_directory='./logs',
    ):
        # Initialize environment
        self.env = env
        self.env.reset()
        self.state_shape = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        # Initialize hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.update_rate = update_rate

        # Initialize Q-network
        self.q_network = QNetwork(self.state_shape, self.action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-6)

        # Initialize replay buffer
        self.memory = ReplayBuffer(memory_size, batch_size)

        # Other variables
        self.timestep = 0
        self.epsilon = 1

        # Logging to tensorboard
        print(f"Run the following command to monitor training: tensorboard --logdir {logging_directory}")
        self.logging_directory = os.path.join(logging_directory, "DQN")
        self.writer = SummaryWriter(log_dir=self.logging_directory)

    def train(self, max_steps):
        state = self.env.reset()
        current_reward = 0
        progress = tqdm(range(max_steps))
        for _ in progress:
            self.timestep += 1

            # Choose and take action
            action = self.act(state, train=True)
            next_state, reward, done, info = self.env.step(action)
            current_reward += reward

            # Save it
            self.memory.add(state, action, reward, next_state, done)

            # Update network when needed
            if (self.timestep % self.update_rate == 0) & (len(self.memory) > self.batch_size):
                loss = self.learn()
                progress.set_description(f"Timestep: {self.timestep}, Loss: {loss:.4f}")
                self.writer.add_scalar('loss', loss, self.timestep)

            # Update epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Update state
            if done:
                self.writer.add_scalar('episode_reward', current_reward, self.timestep)
                # img = np.transpose(state_to_image(self.env.render(mode='array'), show_ball=False), (2,0,1))
                # self.writer.add_image('episode_frame', img, self.timestep)
                state = self.env.reset()
                current_reward = 0
            else:
                state = next_state

            # Logging to tensorboard
            self.writer.add_scalar('epsilon', self.epsilon, self.timestep)

    def learn(self):
        # Sample from memory
        experiences_batch = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences_batch

        # Get the action with max Q-value
        action_values = self.q_network(next_states).detach()
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # If done just use reward, else update Q_target with discounted action values
        q_target = rewards + (self.gamma * max_action_values * (1 - dones))
        q_expected = self.q_network(states).gather(1, actions)

        # Update Q-network
        loss = F.mse_loss(q_target, q_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, state, train=False):
        # Epsilon-greedy action selection
        if train and (random.uniform(0, 1) < self.epsilon):
            return self.env.action_space.sample()
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = self.q_network(state)
            action = np.argmax(action_values.cpu().data.numpy())
            return action

    def save(self, path, save_memory=False, override=False):
        # create save directory
        try:
            os.mkdir(path)
        except FileExistsError:
            if not override:
                raise FileExistsError(f"Save directory {path} already exists. Use override=True to overwrite it.")
        # save q-network weights
        torch.save(self.q_network.state_dict(), os.path.join(path, 'q_network.pth'))
        # save hyperparameters
        with open(os.path.join(path, 'hyperparameters.txt'), 'w') as f:
            f.write(f"gamma: {self.gamma}\n")
            f.write(f"lr: {self.lr}\n")
            f.write(f"batch_size: {self.batch_size}\n")
            f.write(f"epsilon_min: {self.epsilon_min}\n")
            f.write(f"epsilon_decay: {self.epsilon_decay}\n")
            f.write(f"memory_size: {self.memory_size}\n")
            f.write(f"update_rate: {self.update_rate}\n")
        # save memory
        if save_memory:
            self.memory.save(os.path.join(path, 'memory.pkl'))
        else:
            print("Memory not saved. To save it, specify save_memory=True (at the price of a higher disk usage).")
        # save other variables
        torch.save([self.timestep, self.epsilon], os.path.join(path, 'variables.pth'))

    @staticmethod
    def load(path, env):
        # Create agent with env
        agent = DQNAgent(env=env)
        # load q-network weights
        agent.q_network.load_state_dict(torch.load(os.path.join(path, 'q_network.pth')))
        # load hyperparameters
        with open(os.path.join(path, 'hyperparameters.txt'), 'r') as f:
            for line in f:
                key, value = line.split(':')
                setattr(agent, key.strip(), float(value.strip()))
        # load memory
        if os.path.exists(os.path.join(path, 'memory.pkl')):
            agent.memory = ReplayBuffer.load(os.path.join(path, 'memory.pkl'))
        else:
            print("No memory available to load. The experience replay will start empty if you continue training.")
        # load other variables
        agent.timestep, agent.epsilon = torch.load(os.path.join(path, 'variables.pth'))
        # return agent
        return agent
