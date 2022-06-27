import numpy as np


def state_to_image(state, show_ball=True):
    img = state[:, :, :3].astype(np.uint8)  # remove last channel and convert to uint8
    img[:, :, 0] += state[:, :, 3]  # move bricks and paddle to the same channel
    if not show_ball:
        img = img[:, :, 0:1]  # keep only the first channel
    return 255*img

def image_to_state(img):
    s=np.zeros(shape=(10,10,4))
    s[-1,:,0]=img[-1,:,0]
    s[:,:,1:3]=img[:,:,1:]
    s[:-1,:,3]=img[:-1,:,0]
    s=s.astype(np.bool8)
    return s
