import gym
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, Concatenate
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory
from DeepRLmethods import build_DDPG_agent
import random
import pandas as pd

# setting random seed
seed = 1
random.seed(seed)
np.random.seed(seed)

import matplotlib.pyplot as plt

# Setting up the environment
env = gym.make('HalfCheetah-v3')
env_shape = env.observation_space.shape
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]
print(env_shape)
print("HalfCheetah states:")
print("states:", states, "actions:", actions)

# Setting up hyper parameters
lr = 1e-3  # learning rate
window_length = 1  # how many frames take up one state
max_step_trial = 5000  # max amount of steps for each trial
training_steps = 100000  # max amount of training runs

# instantiating agent
ddpg_agent = build_DDPG_agent(env=env)
ddpg_agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])
print('done compiling')

# training the data
print("start training")
fit_hist = ddpg_agent.fit(env,
                          nb_steps=training_steps,
                          visualize=False,
                          verbose=1,
                          nb_max_episode_steps=2000)
# Finally, evaluate our algorithm for 5 episodes.
ddpg_agent.test(env,
                nb_episodes=1,
                visualize=True,
                nb_max_episode_steps=600)

# After training is done, we save the final weights.
ddpg_agent.save_weights('ddpg_{}_weights.h5f'.format("cheetah"), overwrite=True)

# saving training data:
results_dict = {'epochs': fit_hist.epoch,
                'rewards': fit_hist.history['episode_reward']}

df_results = pd.DataFrame(results_dict)
df_results.to_csv('learning_results_ddpg_cheetah.csv')

