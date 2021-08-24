import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


from . import common


def main(debug=False):
    name = ['I', 'SCA', 'tfp']
    suffix = [10, '', '']
    dfs = []
    for n, s in zip(name, suffix):
        df = pd.read_csv(f'results/logk_scale_{n}{s}.csv')
        df['type'] = n
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df['time'] = 1000 * df['time']
    df = df.reset_index()
    print(df)

    fig = common.figure(figsize=(4, 2), box=debug)
    ax = fig.subplots(1, 1)
    sns.lineplot(x='size', y='time', hue='type', hue_order=['I', 'SCA', 'tfp'], data=df, ax=ax)
    ax.set_xscale('log')
    ax.set_ylabel('time (msec)')
    ax.set_ylim(0, 38)
    fig.savefig('figs/fig6.pdf')


if __name__ == '__main__':
    main(debug=False)
