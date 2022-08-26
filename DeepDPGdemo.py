import gym
import numpy as np
from tensorflow.keras.optimizers import Adam
from DeepRLmethods import build_DDPG_agent
# from DeepDPGLearningCheetah import build_DDPG_agent
import matplotlib.pyplot as plt
import pandas as pd

#defining the environment
# env = gym.make('HalfCheetah-v3')
# env_shape = env.observation_space.shape


#building the agent for demo
# demo_agent = build_DDPG_agent(env=env)
# demo_agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])
# demo_agent.load_weights(filepath='ddpg_{}_weights.h5f'.format("cheetah"))
# print('done building agent')
#
# _ = demo_agent.test(env, nb_episodes=10, visualize=True, verbose=1, nb_max_episode_steps=600)

#plotting results
df_runsets = pd.DataFrame(columns=['epochs', 'rewards'])
n_sets = 2
for i in range(n_sets):
    learning_hist = pd.read_csv('learning_results_ddpg_cheetah{}.csv'.format(str(i)))
    df_runsets = pd.concat([df_runsets, learning_hist])
epochs = df_runsets['epochs']
rewards = df_runsets['rewards']


plt.scatter(epochs, rewards, color="red")
plt.show()
