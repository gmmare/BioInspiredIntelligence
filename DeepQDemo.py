import gym
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from DeepRLmethods import build_DQN_agent

import random
import matplotlib.pyplot as plt
import pandas as pd

# defining env
env = gym.make('HalfCheetah-v3')
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]

#loading previously learned model model
trained_model = load_model('dqn_model.h5')
trained_model.summary()

#instantiating the model itself
dqn = build_DQN_agent(trained_model, actions)
dqn.compile(Adam(lr=1e-4), metrics=['mae'])

#demo
_ = dqn.test(env, nb_episodes=10, visualize=True, verbose=1, nb_max_episode_steps=600)

#plotting reward
learning_hist = pd.read_csv('learning_results.csv')
epochs = learning_hist['epochs']
rewards = learning_hist['rewards']
plt.plot(epochs, rewards)
plt.show()
