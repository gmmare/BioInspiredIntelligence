import gym
from rl.agents import DQNAgent
from tensorflow.keras.optimizers import Adam
from DeepQLearningCheetah import build_model, build_agent, params
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

env = gym.make('HalfCheetah-v3')

#defining states and action space
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]

#building a model from functions
model_dqn = load_model('dqn_model.h5f')
# model = build_model(states, actions, window_length=params['window_length'])
# dqn = build_agent(model, actions)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#presenting model
_ = model_dqn.test(env, nb_episodes=5, visualize=True)

#plotting reward
fit_hist = params['history']
epochs = fit_hist.epoch
rewards = fit_hist.history['episode_reward']
plt.plot(epochs, rewards)
plt.show()