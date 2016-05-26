import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2

def plot_simulation_run():
    df = pd.read_csv('softmax_experiment.csv')
    df_reward = df[df['context'] == 'reward'].reset_index()
    df_punishment = df[df['context'] == 'punishment'].reset_index()

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(df_reward['action'])
    ax1.set_xlabel('trial')
    ax1.set_ylabel('action')
    ax1.set_yticks([3, 8, 14, 23])
    ax1.set_ylim(ymin=2, ymax=28)
    ax1.set_title('Reward context')
    df_wins = df_reward.loc[df_reward['reward'] > 0]
    df_lose = df_reward.loc[df_reward['reward'] < 0]
    pos_win = df_wins.index
    pos_lose = df_lose.index
    ax1.eventplot(pos_win, lineoffsets=25.5, linelength=4,
                  linewidths=2)
    ax1.eventplot(pos_lose, lineoffsets=25.5, linelength=4,
                  color='r', linewidths=2)

    ax2.plot(df_punishment['action'])
    ax2.set_xlabel('trial')
    ax2.set_ylabel('action')
    ax2.set_yticks([3, 8, 14, 23])
    ax2.set_ylim(ymin=2, ymax=28)
    ax2.set_title('Punishment context')
    df_wins = df_punishment.loc[df_punishment['reward'] > 0]
    df_lose = df_punishment.loc[df_punishment['reward'] < 0]
    pos_win = df_wins.index
    pos_lose = df_lose.index
    ax2.eventplot(pos_win, lineoffsets=25.5, linelength=4,
                  linewidths=2)
    ax2.eventplot(pos_lose, lineoffsets=25.5, linelength=4,
                  color='r', linewidths=2)

    plt.tight_layout()
    plt.savefig('softmax_experiment.pdf')
    plt.show()
