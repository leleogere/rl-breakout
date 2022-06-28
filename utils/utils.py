import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as JImage
from IPython.display import display
from PIL import Image

from agents.dqn import DQNAgent


def state_to_image(state: np.ndarray, show_ball: bool = True) -> np.ndarray:
    """Convert a state to an image.
    Bring the bricks to the first channel (same as the paddle).
    Function specific to MinAtar/Breakout-v1.

    Parameters
    ----------
    state: The state to convert
    show_ball: Whether to show the ball or not (3 channel image if True, 1 otherwise)

    Returns
    -------
    The image representation of the state
    """
    img = state[:, :, :3].astype(np.uint8)  # remove last channel and convert to uint8
    img[:, :, 0] += state[:, :, 3]  # move bricks and paddle to the same channel
    if not show_ball:
        img = img[:, :, 0:1]  # keep only the first channel
    return 255 * img


def render_env(env: gym.Env) -> None:
    """Needed as env.render() did not work by itself
    
    Parameters
    ----------
    env: The environment to render
    """
    img = state_to_image(env.render('array'))
    plt.imshow(img)
    plt.show()


def render_game(frames: list, fps: int = 20, show_every_x_frames: int = 1, path: str = './game.gif', size: int = 200) -> None:
    """Render a game from a list of states.

    Parameters
    ----------
    frames: The list of states to render
    fps: The number of frames per second of the gif
    show_every_x_frames: The number of frames to skip between each frame in the gif
    path: The path to save the gif to
    size: The size of the image to render
    """
    processed_frames = []
    for frame in frames[::show_every_x_frames]:
        img = Image.fromarray(state_to_image(frame)).resize((size, size), Image.Resampling.NEAREST)
        processed_frames.append(np.asarray(img))
    imageio.mimwrite(path, processed_frames, fps=fps)
    display(JImage(open(path, 'rb').read()))


def image_to_state(img: np.ndarray) -> np.ndarray:
    """Convert an image back to a state (put bricks back to last channel).
    Function specific to MinAtar/Breakout-v1.

    Parameters
    ----------
    img: The image to convert

    Returns
    -------
    The state
    """
    s = np.zeros(shape=(10, 10, 4))
    s[-1, :, 0] = img[-1, :, 0]
    s[:, :, 1:3] = img[:, :, 1:]
    s[:-1, :, 3] = img[:-1, :, 0]
    s = s.astype(np.bool8)
    return s


def play_game(env, agent: DQNAgent | None = None, fps: int = 20, show_every_x_frames: int = 1, path: str = './game.gif', size: int = 200) -> None:
    """Play a game and render it.

    Parameters
    ----------
    env: The environment to play on
    agent: The agent to use (if None, a random policy is used)
    fps: The number of frames per second of the gif
    show_every_x_frames: The number of frames to skip between each frame in the gif
    path: The path to save the gif to
    size: The size of the image to render
    """
    frames = []
    rewards = 0
    state = env.reset()
    choose_action = lambda s: env.action_space.sample() if agent is None else agent.act(s)
    while True:
        state, r, done, _ = env.step(choose_action(state))
        frames.append(state)
        rewards += r
        if done:
            break
    render_game(frames, fps, show_every_x_frames, path, size)
    print(f"Total rewards: {rewards}")
