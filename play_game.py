import argparse
import os
import gym

from utils.utils import play_game
from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent', type=str, default='ddqn', help='Agent choice : DQN/DDQN')
    parser.add_argument('--path_weights', type=str, default=None, help='Path to agent weight')
    parser.add_argument('--gif_name', type=str, default='game.gif', help='GIF file name')
    args = parser.parse_args()

    agent_type = args.agent
    gif_name = args.gif_name
    agent_file = args.path_weights

    env = gym.make('MinAtar/Breakout-v1')
    env.reset()

    saving_dir = './networks_weights/'

    if agent_file:
        agent_path = os.path.join(saving_dir, agent_file + '.zip')
        agent = DQNAgent.load(agent_path)
    else:
        if agent_type == 'dqn':
            print("Load DQN weights")
            agent_file = 'DQN_lr=5e-4_g=0.999_bs=256_ed=0.9995_em=0.005_ms=500000_ts=2e6'
            agent_path = os.path.join(saving_dir, agent_file + '.zip')
            agent = DQNAgent.load(agent_path)

        else:
            print("Load Double DQN weights")
            agent_file = 'DDQN_lr=5e-4_g=0.999_bs=256_ed=0.9995_em=0.005_ms=500000_ts=2e6'
            agent_path = os.path.join(saving_dir, agent_file + '.zip')
            agent = DoubleDQNAgent.load(agent_path)

    play_game(env, agent, path=f"./games/{gif_name}")
