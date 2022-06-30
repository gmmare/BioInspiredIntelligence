import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

'''
=========== project outline: ===========
- Define the gym cheetah environment

- define classes for the project;
    - define agent class

- network itself
    - Bellman eq: f
'''
#check environment
# env_name = "HalfCheetah-v3"
# env = gym.make(env_name)
# print("Observation space:", env.observation_space)
# print("Action space:", env.action_space)
#
# class DeepQlearnAgent():
#     def __init__(self, env):
#         self.obs_space = env.observation_space
#         self.act_space = env.action_space
#
#     def get_action(self):
#         s = np.random.default_rng().uniform(0,1,6)
#         return s
#
# agent = DeepQlearnAgent(env)
# env.reset()
#
# action = agent.get_action()
# for i in range(1000):
#     action = np.random.default_rng().uniform(0,1,6)
#     observation, reward, done, info = env.step(action)
#     # print(i, env.time_step_spec().observation())
#     env.render()
#     print(observation, reward, done, info)
#
# env.close()

# Setting up the environment
env = gym.make('HalfCheetah-v3')
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]
print(states, actions)

#building the model definition
L1 = 30
L2 = 30
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(L1, activation='relu'))
    model.add(Dense(L2, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

#instantiate model
model = build_model(states, actions)
model.summary() #showing model setup

#creating dqn agent
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model,
                   memory=memory,
                   policy=policy,
                   nb_actions=actions,
                   nb_steps_warmup=10,
                   target_model_update=1e-2)
    return dqn

#instantiating agent
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#training the data
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

#testing the agent (already learnt by now)
_ = dqn.test(env, nb_episodes=15, visualize=True)
dqn.save_weights('dqn_weights.h5f', overwrite=True)
