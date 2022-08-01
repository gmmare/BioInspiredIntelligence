import gym
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
import random
import matplotlib.pyplot as plt
import pandas as pd

# Setting up hyperparameters
lr = 1e-3 #learning rate
limit_test = 50000      #number of frames for each trail
window_length = 3       #how many frames take up one state
L1 = 80                 #dense layer nodes: layer 1
L2 = 60                 #dense layer nodes: layer 2
L3 = 60                 #dense layer nodes: layer 3
max_explore = 1         #value for exploration (eps)
min_explore = 0.1
max_step_trial = 10000  #max amount of steps for each trial
warm_up_trials = 200    # amount of trials
training_steps = 10000  # max amount of training runs

env = gym.make('HalfCheetah-v3')
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]
#testing agent:
# action_input = np.random.uniform(low=0., high=2, size=actions)
# env.reset()
# env.step(action=action_input)

#instantiate model
trained_model = load_model('dqn_model.h5')
trained_model.summary()
# dqn_model.get_weights()

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=max_explore,
                                  value_min=min_explore,
                                  value_test=.05,
                                  nb_steps=max_step_trial)
    memory = SequentialMemory(limit=limit_test, window_length=window_length)
    dqn = DQNAgent(model=model,
                   memory=memory,
                   policy=policy,
                   nb_actions=actions,
                   nb_steps_warmup=30,
                   target_model_update=1e-2)
    return dqn

#instantiating the model itself
dqn = build_agent(trained_model, actions)
dqn.compile(Adam(lr=lr), metrics=['mae'])
#demo
_ = dqn.test(env, nb_episodes=10, visualize=True, verbose=1)

#plotting reward
learning_hist = pd.read_csv('learning_results.csv')
epochs = learning_hist['epochs']
rewards = learning_hist['rewards']
plt.plot(epochs, rewards)
plt.show()