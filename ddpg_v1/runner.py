import gym
import numpy as np
import matplotlib.pyplot as plt
# from ddpg import ddpg
from ddpg_class import ddpg

def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
    return y

# returns, q_losses,mu_losses = ddpg(lambda : gym.make('Pendulum-v0'),num_train_episodes=50)
ddpg_net = ddpg(env_fn=gym.make('Pendulum-v0'), num_train_episodes=1)
returns, q_losses,mu_losses = ddpg_net.train_ddpg()
ddpg_net.SaveNet()
ddpg_net.Demo()


plt.plot(returns)
plt.plot(smooth(np.array(returns)))
plt.title("Train returns")
plt.show()

# plt.plot(test_returns)
# plt.plot(smooth(np.array(test_returns)))
# plt.title("Test returns")
# plt.show()

plt.plot(q_losses)
plt.title('q_losses')
plt.show()

plt.plot(mu_losses)
plt.title('mu_losses')
plt.show()