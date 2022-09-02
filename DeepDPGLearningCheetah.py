import gym
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from DeepRLmethods import build_DDPG_agent, sensitivity_analysis
import pandas as pd
import random

# setting random seed
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# save weights
save_weights = False
save_data = False

# Setting up the environment
env = gym.make('HalfCheetah-v3')
env_shape = env.observation_space.shape
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]
print(env_shape)
print("HalfCheetah states:")
print("states:", states, "actions:", actions)

# Setting up hyper parameters
window_length = 1  # how many frames take up one state
max_step_trial = 5000  # max amount of steps for each trial
training_steps = 100000  # max amount of training runs
gamma = 0.99
sigma = 0.15
lr_actor = 1e-4

# instantiating agent
ddpg_agent = build_DDPG_agent(env=env, gamma=gamma, sigma=sigma)
ddpg_agent.compile([Adam(lr=lr_actor), Adam(lr=1e-3)], metrics=['mae'])
print('done compiling')

# training the data
print("start training")
fit_hist = ddpg_agent.fit(env,
                          nb_steps=training_steps,
                          visualize=False,
                          verbose=1,
                          nb_max_episode_steps=2000)

# Saving weights
if save_weights:
    ddpg_agent.save_weights('WorkingNets/ddpg_cheetah_weights.h5f', overwrite=True)

if save_data:
    # saving training data:
    results_dict = {'epochs': fit_hist.epoch,
                    'rewards': fit_hist.history['episode_reward']}

    df_results = pd.DataFrame(results_dict)
    n_file = 7
    file_name = 'ddpg_results/learning_results' + str(n_file) + '.csv'
    df_results.to_csv(file_name)

# sensitivity analysis
sensitivity_analysis(env=env, agent=ddpg_agent, n_trials=40)
print("lr = ", lr_actor)
print("gamma = ", gamma)
print("sigma = ", sigma)
