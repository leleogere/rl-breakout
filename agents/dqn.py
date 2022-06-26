import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import os
from tqdm import tqdm
import shutil
from zipfile import ZipFile
from datetime import datetime

from utils.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer
from utils.utils import state_to_image

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
            agent_name='DQN',
            log_images=True,
            checkpoint_directory='./networks_weights',
    ):
        self.agent_name = agent_name
        self.id = self.agent_name + '_' + datetime.now().strftime('%Y%m%d-%H%M%S')

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
        self.checkpoint_path = os.path.join(checkpoint_directory, f"Checkpoint_{self.id}.zip")
        self.best_reward = -np.inf
        self.timestep = 0
        self.epsilon = 1

        # Logging to tensorboard
        print(f"Run the following command to monitor training: tensorboard --logdir {logging_directory}")
        self.logging_directory = os.path.join(logging_directory, agent_name)
        self.writer = SummaryWriter(log_dir=self.logging_directory)
        self.log_images = log_images

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
                progress.set_description(f"Timestep: {self.timestep}, Loss: {loss:02.4f}")
                self.writer.add_scalar('loss', loss, self.timestep)

            # Update epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Update state
            if done:
                self.writer.add_scalar('episode_reward', current_reward, self.timestep)
                if current_reward > self.best_reward:
                    self.writer.add_scalar('best_reward', current_reward, self.timestep)
                    self.save(self.checkpoint_path, save_memory=True, override=True)
                    self.best_reward = current_reward
                if self.log_images:
                    img = np.transpose(state_to_image(next_state, show_ball=False), (2,0,1))
                    self.writer.add_image('last_episode_frame', img, self.timestep)
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

    def save(self, path, save_memory=True, override=False):
        if os.path.exists(path) and not override:
            raise FileExistsError(f"Agent {path} already exists. Use override=True to overwrite it.")
        tmp_path = os.path.join(path + '_' + datetime.now().strftime('%Y%m%d-%H%M%S'), "agent.pt")
        os.makedirs(os.path.dirname(tmp_path))
        writer_backup = self.writer
        memory_backup = self.memory
        try:
            self.writer = None
            if not save_memory:
                print("Memory not saved. To save it, specify save_memory=True (at the price of a higher disk usage).")
                self.memory = None
            torch.save(self, tmp_path)
            with ZipFile(path, 'w') as zip_file:
                zip_file.write(tmp_path, arcname=os.path.basename(tmp_path))
            shutil.rmtree(os.path.dirname(tmp_path))
        finally:
            self.writer = writer_backup
            if not save_memory:
                self.memory = memory_backup

    @staticmethod
    def load(path, device='cuda') ->  'DQNAgent':
        tmp_path = path + '_' + datetime.now().strftime('%Y%m%d-%H%M%S')
        shutil.unpack_archive(path, tmp_path)
        try:
            agent = torch.load(tmp_path + "/agent.pt", map_location=torch.device(device))
            if not hasattr(agent, 'writer') or agent.writer is None:
                agent.writer = SummaryWriter(log_dir=agent.logging_directory)
            if agent.memory is None:
                print("No memory available to load. The experience replay will start empty if you continue training.")
                agent.memory = ReplayBuffer(agent.memory_size, agent.batch_size)
        finally:
            shutil.rmtree(tmp_path)
        return agent
