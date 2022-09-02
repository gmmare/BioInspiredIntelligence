import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#getting number of files
count = 0
series_name = "dqn"

dir_path = series_name + '_results'
for path in os.scandir(dir_path):
    if path.is_file():
        count += 1
print('file count:', count)

#setting up data structure
df_runsets = pd.DataFrame(columns=['epochs', 'rewards'])
df_variance = pd.DataFrame()

for i in range(count):
    filename = series_name + '_results/learning_results{}.csv'.format(str(i))
    learning_hist = pd.read_csv(filename)
    df = learning_hist[['epochs', 'rewards']]
    df_runsets = pd.concat([df_runsets, df])

    #getting variance
    df_var = learning_hist[['rewards']]
    df_variance = pd.concat([df_variance, df_var], axis=1)

#calculating variance

df_var_sqrt = df_variance.var(axis=1, ddof=0)**(1/2)
df_var_sqrt = df_var_sqrt.to_frame('var')

#prepping range
epochs = df_runsets['epochs']
rewards = df_runsets['rewards']

sns.set_theme(style="darkgrid")

#plotting data
fig1 = sns.lineplot(x="epochs", y="rewards", data=df_runsets)
fig1.set(xlabel='Epochs', ylabel='Rewards')
plt.show()

fig2 = sns.regplot(x=df_var_sqrt.index, y='var', data=df_var_sqrt, order=3)
fig2.set(xlabel='Epochs', ylabel='Reward variance')
plt.show()