import gym
from tensorflow.keras.optimizers import Adam
from DeepRLmethods import build_DDPG_agent, sensitivity_analysis
from gym.wrappers import Monitor

#set to true if you want video recording, for sensitivity analysis, set video to False
video = False

if video:
    env = Monitor(gym.make('HalfCheetah-v3'), './video_ddpg', force=True)
    display_video = False
else:
    env = gym.make('HalfCheetah-v3')
    display_video = True
env_shape = env.observation_space.shape


#building the agent for demo
ddpg = build_DDPG_agent(env=env)
ddpg.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])
ddpg.load_weights(filepath='WorkingNets/ddpg_cheetah_weights.h5f')
print('done building agent')

#comment this when doing sensitivity analysis quick results
# _ = demo_agent.test(env, nb_episodes=1, visualize=display_video, verbose=1, nb_max_episode_steps=600)

#sensitivity analysis results computation
sensitivity_analysis(env=env, agent=ddpg, n_trials=40)