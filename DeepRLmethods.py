import gym
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, Concatenate
from rl.agents import DDPGAgent, DQNAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
import random

#quick net building
def build_model(states, actions, window_length, L):
    """
    :param states:      state space
    :param actions:     action space
    :return:            keras deeplearning model
    """
    model = Sequential()
    model.add(Flatten(input_shape=(window_length,states)))

    for i in L:
        model.add(Dense(i, activation='relu'))         #default is relu

    model.add(Dense(actions, activation='tanh'))
    return model

#defining DQN agent
def build_DQN_agent(model, actions):
    max_explore = 1         #value for exploration (eps)
    min_explore = 0.1
    max_step_trial = 10000  #max amount of steps for each trial
    window_length = 3
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=max_explore,
                                  value_min=min_explore,
                                  value_test=.05,
                                  nb_steps=max_step_trial)
    memory = SequentialMemory(limit=100000,
                              window_length=window_length)
    dqn = DQNAgent(model=model,
                   memory=memory,
                   policy=policy,
                   nb_actions=actions,
                   nb_steps_warmup=30,
                   target_model_update=1e-2)
    return dqn

#DDPG agent
def build_DDPG_agent(env):
    '''

    :param env:
    :return:
    '''


    states = env.observation_space.shape[0]
    actions = env.action_space.shape[0]

    # setting up training layers
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    action_input = Input(shape=(actions,), name='action_input')

    #making the actor model
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(400))
    actor.add(Activation('relu'))
    actor.add(Dense(300))
    actor.add(Activation('relu'))
    actor.add(Dense(actions))
    actor.add(Activation('tanh'))
    print(actor.summary())

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

    # setting up replay memory
    memory = SequentialMemory(limit=100000,
                              window_length=1)

    # setting up random input noise
    random_process = OrnsteinUhlenbeckProcess(size=actions, theta=.15, mu=0., sigma=.1)

    # instantiating agent
    agent = DDPGAgent(nb_actions=actions,
                      nb_steps_warmup_actor=1000,
                      nb_steps_warmup_critic=1000,
                      actor=actor,
                      critic=critic,
                      critic_action_input=action_input,
                      random_process=random_process,
                      memory=memory)
    return agent