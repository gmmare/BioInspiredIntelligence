import gym
import numpy as np
from tensorflow.keras.models import Sequential, save_model, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, Concatenate
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory, Memory
import random
import pandas as pd

# setting random seed
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
lr = 1e-3  # learning rate
window_length = 1  # how many frames take up one state
max_step_trial = 5000  # max amount of steps for each trial
training_steps = 100000  # max amount of training runs

# setting up layers for training
nb_actions = actions
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)

# actor_layers = [500, 500, 200]
# critic_layers = [400, 200]
#
# # defining actor model
# actor = Sequential()
# actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#
# for nodes in actor_layers:
#     actor.add(Dense(nodes))
#     actor.add(Activation('relu'))
#
# actor.add(Dense(nb_actions))
# actor.add(Activation('tanh'))
# print(actor.summary())
#
# action_input = Input(shape=(nb_actions,), name='action_input')
# observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
# flattened_observation = Flatten()(observation_input)
#
# # defining critic model
# # first concatenate actor output and states
# x = Dense(400)(flattened_observation)
# x = Activation('relu')(x)
# x = Concatenate()([x, action_input])
#
# for nodes in critic_layers:  # assembling layers
#     x = Dense(nodes)(x)
#     x = Activation('relu')(x)
#
# x = Dense(1)(x)  # adding output layers
# x = Activation('tanh')(x)
# critic = Model(inputs=[action_input, observation_input], outputs=x)
# print(critic.summary())

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(400))
actor.add(Activation('relu'))
actor.add(Dense(300))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(400)(flattened_observation)
x = Activation('relu')(x)
x = Concatenate()([x, action_input])
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


# creating ddpg agent
def build_DDPG_agent(actor_net, critic_net, actions):
    memory = SequentialMemory(limit=100000,
                              window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=actions, theta=.15, mu=0., sigma=.1)
    agent = DDPGAgent(nb_actions=actions,
                      nb_steps_warmup_actor=1000,
                      nb_steps_warmup_critic=1000,
                      actor=actor_net,
                      critic=critic_net,
                      critic_action_input=action_input,
                      random_process=random_process,
                      memory=memory)
    return agent


# instantiating agent
ddpg_agent = build_DDPG_agent(actor_net=actor,
                              critic_net=critic,
                              actions=actions)
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
                nb_episodes=5,
                visualize=True,
                nb_max_episode_steps=600)

# After training is done, we save the final weights.
ddpg_agent.save_weights('ddpg_{}_weights.h5f'.format("cheetah"), overwrite=True)

# # testing the agent on a new set
# scores = ddpg_agent.test(env, nb_episodes=20, visualize=False)
# print(np.mean(scores.history['episode_reward']))
#
# # demo
# _ = ddpg_agent.test(env, nb_episodes=1, visualize=True)
#
#
#
# saving training data:
results_dict = {'epochs': fit_hist.epoch,
                'rewards': fit_hist.history['episode_reward']}

df_results = pd.DataFrame(results_dict)
df_results.to_csv('learning_results_ddpg_cheetah.csv')

# # saving the model
# ddpg_agent.save('ddpg_model.h5')
# ddpg_agent.save_weights('dqn_weights.h5f', overwrite=True)
