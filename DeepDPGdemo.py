import gym
import numpy as np
from tensorflow.keras.optimizers import Adam
from DeepRLmethods import build_DDPG_agent
# from DeepDPGLearningCheetah import build_DDPG_agent


import random
import pandas as pd

#defining the environment
env = gym.make('HalfCheetah-v3')
env_shape = env.observation_space.shape


#building the agent for demo
demo_agent = build_DDPG_agent(env=env)
demo_agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])
demo_agent.load_weights(filepath='ddpg_{}_weights.h5f'.format("cheetah"))
print('done building agent')

_ = demo_agent.test(env, nb_episodes=10, visualize=True, verbose=1, nb_max_episode_steps=600)
