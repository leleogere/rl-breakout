import argparse
import os
import gym

from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from utils.utils import play_game

if __name__ == '__main__':
    env = gym.make('MinAtar/Breakout-v1')
    env.reset()

    parser = argparse.ArgumentParser()

    parser.add_argument('--agent', type=str, default='ddqn', help='Agent choice : DQN/DDQN')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum of epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.999, help='Decay of epsilon')
    parser.add_argument('--memory_size', type=int, default=100_000, help='Memory size')
    parser.add_argument('--nb_episodes', type=int, default=5000, help='number of episodes')
    parser.add_argument('--play_game', type=bool, default=False, help='Play a game after the training')

    args = parser.parse_args()

    agent_type = args.agent
    gamma = args.gamma
    lr = args.lr
    batch_size = args.batch_size
    epsilon_min = args.epsilon_min
    epsilon_decay = args.epsilon_decay
    memory_size = args.memory_size
    nb_episodes = args.nb_episodes
    play_a_game = args.play_game

    saving_dir = './networks_weights/'

    if agent_type == 'dqn':
        agent = DQNAgent(
            env,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            memory_size=memory_size,
            checkpoint_directory=saving_dir,
        )

    else:
        agent = DoubleDQNAgent(
            env,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            memory_size=memory_size,
            checkpoint_directory=saving_dir,
        )

    agent.train(nb_episodes)

    agent_path = os.path.join(saving_dir, agent_type + '.zip')
    agent.save(agent_path, override=True, save_memory=True)

    if play_a_game:
        play_game(env, agent, path=f"./games/{agent_type}.gif")
