import gym
import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
import random
import pandas as pd

#setting random seed
seed = 1
random.seed(seed)
np.random.seed(seed)

import matplotlib.pyplot as plt

# Setting up the environment
env = gym.make('HalfCheetah-v3')
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]
print("HalfCheetah states:")
print("states:", states, "actions:", actions)

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

def build_model(states, actions, window_length):
    """
    :param states:      state space
    :param actions:     action space
    :return:            keras deeplearning model
    """
    model = Sequential()
    model.add(Flatten(input_shape=(window_length,states)))
    model.add(Dense(L1, activation='relu'))         #default is relu
    model.add(Dense(L2, activation='relu'))
    model.add(Dense(L3, activation='relu'))
    model.add(Dense(actions, activation='tanh'))
    return model

#instantiate model
model = build_model(states, actions, window_length)
model.summary() #showing model setup

#creating dqn agent
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

#instantiating agent
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=lr), metrics=['mae'])

#training the data
print("start training")
fit_hist = dqn.fit(env, nb_steps=training_steps, visualize=False, verbose=1)

#testing the agent on a new set
scores = dqn.test(env, nb_episodes=20, visualize=False)
print(np.mean(scores.history['episode_reward']))

#demo
# _ = dqn.test(env, nb_episodes=1, visualize=True)
#closing environment


#saving training data:
results_dict = {'epochs':fit_hist.epoch,
                'rewards':fit_hist.history['episode_reward']}

df_results = pd.DataFrame(results_dict)
df_results.to_csv('learning_results.csv')

#saving the model
model.save('dqn_model.h5')
dqn.save_weights('dqn_weights.h5f', overwrite=True)



