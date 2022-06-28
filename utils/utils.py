import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Image as JImage


def state_to_image(state, show_ball=True):
    img = state[:, :, :3].astype(np.uint8)  # remove last channel and convert to uint8
    img[:, :, 0] += state[:, :, 3]  # move bricks and paddle to the same channel
    if not show_ball:
        img = img[:, :, 0:1]  # keep only the first channel
    return 255 * img


def render_env(env):
    """Needed as env.render() did not work by itself"""
    img = state_to_image(env.render('array'))
    plt.imshow(img)
    plt.show()


def render_game(frames, fps=20, show_every_x_frames=1, path='games.gif', size=200):
    processed_frames = []
    for frame in frames[::show_every_x_frames]:
        img = Image.fromarray(state_to_image(frame)).resize((size, size), Image.Resampling.NEAREST)
        processed_frames.append(np.asarray(img))
    imageio.mimwrite(path, processed_frames, fps=fps)
    display(JImage(open(path, 'rb').read()))


def image_to_state(img):
    s = np.zeros(shape=(10, 10, 4))
    s[-1, :, 0] = img[-1, :, 0]
    s[:, :, 1:3] = img[:, :, 1:]
    s[:-1, :, 3] = img[:-1, :, 0]
    s = s.astype(np.bool8)
    return s


def play_game(env, agent=None, fps=20, show_every_x_frames=1, path='games.gif', size=200):
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
