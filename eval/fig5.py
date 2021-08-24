import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


from . import common


def v_loc(x):
    return 40*np.log10(x + 1)


def x_loc(x):
    return 40*(np.log10(x) + 1)


def main(debug=False):
    thr_v = 35.0
    thr_x = lambda v: 1.7 + 0.08 * v

    df = []
    for name in ['S', 'C', 'A', 'I10', 'tfp']:
        prec = pd.read_csv(f'data/logk_prec_{name}.csv')
        prec = prec.groupby(['v', 'x'])['log_err'].mean()
        prec.name = f'prec_{name}'
        time = pd.read_csv(f'data/logk_time_{name}.csv')
        time = time.groupby(['v', 'x'])['time'].mean()
        time = 1000 * time
        time.name = f'time_{name}'
        df += [prec, time]
    df = pd.concat(df, axis=1)
    v, x = [np.array(z) for z in zip(*df.index)]
    for t in ['prec', 'time']:
        df[f'{t}_SCA'] = df[f'{t}_S']
        df.loc[x < thr_x(v), f'{t}_SCA'] = df[f'{t}_S']
        df.loc[x >= thr_x(v), f'{t}_SCA'] = df[f'{t}_C']
        df.loc[v >= thr_v, f'{t}_SCA'] = df[f'{t}_A']

    df['diff_prec'] = df['prec_SCA'] - df['prec_I10']
    df['diff_time'] = df['time_SCA'] - df['time_I10']

    v, x = zip(*df.index)
    df['v'] = v
    df['x'] = x
    df1 = df[['v', 'x', 'prec_I10', 'prec_SCA', 'prec_tfp']].copy()
    df1.rename(columns=dict(prec_I10='I', prec_SCA='SCA', prec_tfp='tfp'), inplace=True)
    df1 = df1.melt(id_vars=['v','x'])
    df1.rename(columns=dict(variable='type', value='prec'), inplace=True)
    df2 = df[['v', 'x', 'time_I10', 'time_SCA', 'time_tfp']].copy()
    df2.rename(columns=dict(time_I10='I', time_SCA='SCA', time_tfp='tfp'), inplace=True)
    df2 = df2.melt(id_vars=['v','x'])
    df2.rename(columns=dict(variable='type', value='time'), inplace=True)

    type_cmap = ListedColormap(['red', 'blue', 'green'])
    type_cmap.set_under('white')
    name = [['diff_prec', 'prec_SCA'], ['diff_time', 'time_SCA']]
    #pos = [[[0.1, 0.85], [0.85, 0.1]], [[0.1, 0.1], [0.1, 0.85]]]
    vmin = [[-1.0, 0], [-10, 0]]
    vmax = [[+1.0, 2.8], [10, 28]]
    cmap = [['bwr', 'Reds'], ['bwr', 'Blues']]

    fig = common.figure(figsize=(5.5, 4), box=debug)
    ax = fig.subplots(
        2, 2, sharex='col',
        #gridspec_kw=dict(width_ratios=(1,1,0.15)),
    )
    #ax[0, 2].set_visible(False)
    #ax[1, 2].set_visible(False)
    #ax[0, 2] = fig.add_axes([0.93, 0.1, 0.02, 0.4])
    #ax[1, 2] = fig.add_axes([0.93, 0.55, 0.02, 0.4])
    vticks = [0, 1, 5, 10, 50]
    xticks = [0.1, 0.5, 1, 5, 10, 50]

    for i in range(2):
        for j in [0]:
            hm = df[name[i][j]].unstack(0)
            sns.heatmap(hm, vmin=vmin[i][j], vmax=vmax[i][j], cmap=cmap[i][j], ax=ax[i, j])
            v = np.linspace(0, thr_v, 100)
            x = thr_x(v)
            v = v_loc(v)
            x = x_loc(x)
            ax[i, j].plot(v, x, c='k')
            ax[i, j].plot([v_loc(thr_v), v_loc(thr_v)], [x_loc(0.1), x_loc(10**2.1)], c='k')
            ax[i, j].invert_yaxis()
            #ax[i, j].text(*pos[i][j], name[i][j], transform=ax[i, j].transAxes)
            ax[i, j].set_xticks([v_loc(v) for v in vticks])
            ax[i, j].set_xticklabels([f"${k}$" for k in vticks], rotation=0)
            ax[i, j].xaxis.set_ticks_position('both')
            ax[i, j].set_yticks([x_loc(x) for x in xticks])
            ax[i, j].set_yticklabels([f"${k}$" for k in xticks])
            ax[i, j].yaxis.set_ticks_position('both')
            if i == 1:
                ax[i, j].set_xlabel('$v$')
            else:
                ax[i, j].set_xlabel('')
            if j == 0:
                ax[i, j].set_ylabel('$x$')
            else:
                ax[i, j].set_ylabel('')
    sns.boxenplot(x='type', y='prec', data=df1, ax=ax[0, 1])
    ax[0, 1].xaxis.label.set_visible(False)
    sns.boxenplot(x='type', y='time', data=df2, ax=ax[1, 1])
    ax[1, 1].set_ylim(0, 35)
    #cbar = ax[0, 0].collections[0].colorbar
    #cbar.set_ticks([0, 10, 20])
    #cbar.set_ticklabels([f'${{{l}}}$' for l in [0, 10, 20]])

    fig.savefig('figs/fig5.pdf')


if __name__ == '__main__':
    main(debug=False)
