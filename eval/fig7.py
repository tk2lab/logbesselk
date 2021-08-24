import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from . import common


def main(debug=False):
    fig = common.figure(figsize=(5.5, 2), box=debug)
    ax = fig.subplots(
        1, 2,
        gridspec_kw=dict(width_ratios=(0.33,1.)),
    )

    df = pd.read_csv('results/log_dkdv_prec_I10.csv')
    df['type'] = 'I'
    ax[0] = sns.boxenplot(x='type', y='log_err', data=df, ax=ax[0], k_depth='full', color='white')
    ax[0].text(0.1, 0.9, 'a', transform=ax[0].transAxes)
    ax[0].set_ylabel('err ($\log (\Delta/\epsilon + 1)$)')

    name = ['I', 'SCA', 'tfp']
    suffix = ['10', '', '']
    dfs = []
    for n, s in zip(name, suffix):
        df = pd.read_csv(f'results/log_dkdx_prec_{n}{s}.csv')
        df['type'] = n
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    ax[1] = sns.boxenplot(x='type', y='log_err', data=df, ax=ax[1], color='white')
    ax[1].text(0.05, 0.9, 'b', transform=ax[1].transAxes)
    ax[1].set_ylabel('err ($\log (\Delta/\epsilon + 1)$)')

    for i in range(2):
        for c in ax[i].collections[1::2]:
            plt.setp(c, color='k')

    fig.savefig('figs/fig7.pdf')


if __name__ == '__main__':
    main(debug=False)
