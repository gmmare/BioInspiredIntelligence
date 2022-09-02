import gym
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from DeepRLmethods import build_DQN_agent, sensitivity_analysis
from gym.wrappers import Monitor

#set to true if you want video recording
video = False

if video:
    env = Monitor(gym.make('HalfCheetah-v3'), './video_dqn', force=True)
    display_video = False
else:
    env = gym.make('HalfCheetah-v3')
    display_video = True

#getting states
states = env.observation_space.shape[0]
actions = env.action_space.shape[0]

#loading previously learned model model
trained_model = load_model('WorkingNets/dqn_model.h5')
trained_model.summary()

#instantiating the model itself
dqn = build_DQN_agent(trained_model, actions)
dqn.compile(Adam(lr=1e-4), metrics=['mae'])


#demo
check = dqn.test(env, nb_episodes=1, visualize=display_video, verbose=1, nb_max_episode_steps=600)

#sensitivity analysis results computation
# sensitivity_analysis(env=env, agent=dqn, n_trials=40)