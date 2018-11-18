'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import gym
import cv2
from this_util import *
# from model import make_model
# import tensorflow as tf
# tf.enable_eager_execution()

def ss(s=''):
    print('- Message -' * 6)
    print(s)
    print('- End -' * 10)
    halt_massage = '*HALT* *HALT* *HALT* *HALT* *HALT* *HALT* *HALT* *HALT* from "' + __file__ + '"'
    assert False, halt_massage
def image_pre_process(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame

MAX_FRAMES = 10000  # max length of carracing
MAX_TRIALS = 1000  # just use this to extract one trial.
is_render = False
render_mode = False  # for debugging.
# ROOT_PATH = '/media/ray/SSD/workspace/python/dataset/world_model_attempt'
DIR_NAME = 'record'
file_path = os.path.join(ROOT_PATH, DIR_NAME)
# ss('hi')
if not os.path.exists(file_path):
    os.makedirs(file_path)
# ss('ha')
# model = make_model(load_model=False)
env = gym.make('Pong-v0')

total_frames = 0
# while True:
#     print(np.random.choice([2,3]))
# model.make_env(render_mode=render_mode, full_episode=True)
for trial in range(MAX_TRIALS):  # 200 trials per worker
    try:
        random_generated_int = random.randint(0, 2 ** 31 - 1)
        # print(random_generated_int)
        # ss('random')
        filename = file_path + "/" + str(random_generated_int) + ".npz"
        # print
        recording_obs = []
        recording_action = []

        np.random.seed(random_generated_int)
        # model.env.seed(random_generated_int)

        # random policy
        # model.init_random_model_params(stdev=np.random.rand() * 0.01)

        # model.reset()
        # obs = model.env.reset()  # pixels
        # obs = 1
        # print(obs.shape) # 96x96x3 # fixed in env.py, obs is 64x64x3
        # ss('shape')
        # render_mode = False
        obs = env.reset()
        for frame in range(MAX_FRAMES):
            if is_render:
                env.render()
            # print(a.shape) # if rgb_array, return 400x600x3
            # ss('render mode')
            recording_obs.append(obs)

            # z, mu, logvar = model.encode_obs(obs)
            # ss('wait for encode obs')
            # action = model.get_action(z)
            # ss('waiting getting action')
            action = np.random.choice([2, 3])
            recording_action.append(action)
            obs, reward, done, info = env.step(action)

            if done:
                break

        total_frames += (frame + 1)
        print("dead at", frame + 1, "total recorded frames for this worker", total_frames)
        recording_obs = np.array(recording_obs, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.uint8)
        np.savez_compressed(filename, obs=recording_obs, action=recording_action)
    except gym.error.Error:
        print("stupid gym error, life goes on")
        env.close()
        # make_env(render_mode=render_mode)
        continue
env.close()
