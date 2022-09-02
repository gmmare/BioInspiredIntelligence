import gym
from tensorflow.keras.optimizers import Adam
from DeepRLmethods import build_model, build_DQN_agent, sensitivity_analysis
import pandas as pd
import numpy as np
import random

#setting random seed
seed = 1
random.seed(seed)
np.random.seed(seed)

#save weights and data
save_weights = False
save_data = False


# Setting up the environment
env = gym.make('HalfCheetah-v3')
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]
print("HalfCheetah states:")
print("states:", states, "actions:", actions)


# Setting up hyperparameters
limit_test = 5000      #number of frames for each trail
window_length = 3       #how many frames take up one state
L1 = 80                 #dense layer nodes: layer 1
L2 = 60                 #dense layer nodes: layer 2
L3 = 60                 #dense layer nodes: layer 3
max_explore = 0.8         #value for exploration (eps)
min_explore = 0.1
max_step_trial = 10000  #max amount of steps for each trial
warm_up_trials = 200        # amount of trials
training_steps = 100000     # max amount of training runs
gamma = 0.8              #discount rate
lr = 1e-5
L = [L1, L2, L3]


#instantiate model
model = build_model(states, actions, window_length, L)
model.summary()         #showing model setup

#instantiating agent
dqn = build_DQN_agent(model, actions, gamma=gamma, max_explore=max_explore)
dqn.compile(Adam(lr=lr), metrics=['mae'])


#training the data
print("start training")
fit_hist = dqn.fit(env,
                   nb_steps=training_steps,
                   visualize=False,
                   nb_max_episode_steps=2000,
                   verbose=1)

#testing the agent on a new set
# scores = dqn.test(env, nb_episodes=10, visualize=True)
# print(np.mean(scores.history['episode_reward']))

if save_data:
    #saving training data:
    results_dict = {'epochs':fit_hist.epoch,
                    'rewards':fit_hist.history['episode_reward']}

    df_results = pd.DataFrame(results_dict)
    n_file = 9
    file_name = 'dqn_results/learning_results' + str(n_file) + '.csv'
    df_results.to_csv(file_name)
    print("data saved")

#saving the model
if save_weights:
    model.save('WorkingNets/dqn_model.h5')
    dqn.save_weights('WorkingNets/dqn_weights.h5f', overwrite=True)
    print("weights saved")

#sensitivity analysis
sensitivity_analysis(env=env, agent=dqn, n_trials=40)
