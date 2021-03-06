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
    thr_v = 25.0
    thr_x = lambda v: 1.6 + 0.5 * np.log(v + 1)

    name = ['I', 'A', 'S', 'C', 'SCA']
    suffix = ['', '', '', '', '']
    df = []
    for n, s in zip(name, suffix):
        prec = pd.read_csv(f'results/logk_prec_{n}{s}.csv')
        prec = prec.groupby(['v', 'x'])['log_err'].mean()
        prec.name = f'prec_{n}'
        time = pd.read_csv(f'results/logk_time_{n}{s}.csv')
        time = time.groupby(['v', 'x'])['time'].mean()
        time = 1000 * time
        time.name = f'time_{n}'
        df += [prec, time]
    df = pd.concat(df, axis=1)
    v, x = [np.array(z) for z in zip(*df.index)]
    for t in ['prec', 'time']:
        df[f'{t}_SCA'] = df[f'{t}_S']
        df.loc[x < thr_x(v), f'{t}_SCA'] = df[f'{t}_S']
        df.loc[x >= thr_x(v), f'{t}_SCA'] = df[f'{t}_C']
        df.loc[v >= thr_v, f'{t}_SCA'] = df[f'{t}_A']

    df['type_prec'] = -1 
    df['min_prec'] = np.inf
    df['min_time'] = np.inf
    for i, name in enumerate(['S', 'C', 'A']):
        cond = df[f'prec_{name}'] < df['min_prec']
        df.loc[cond, 'type_prec'] = i
        df.loc[cond, 'min_prec'] = df.loc[cond, f'prec_{name}']
        df.loc[cond, 'min_time'] = df.loc[cond, f'time_{name}']

    df['type_time'] = -1 
    df['min_prec'] = np.inf
    df['min_time'] = np.inf
    for i, name in enumerate(['S', 'C', 'A']):
        cond = (df[f'prec_{name}'] < 1.) & (df[f'time_{name}'] < df['min_time'])
        df.loc[cond, 'type_time'] = i
        df.loc[cond, 'min_prec'] = df.loc[cond, f'prec_{name}']
        df.loc[cond, 'min_time'] = df.loc[cond, f'time_{name}']

    #type_cmap = ListedColormap(['magenta', 'blue', 'green', 'cyan'])
    #type_cmap.set_under('white')
    name = [['type_prec', 'prec_SCA'], ['type_time', 'time_SCA']]
    #pos = [[[0.1, 0.85], [0.85, 0.1]], [[0.1, 0.1], [0.1, 0.85]]]
    vmin = [[-1.5, 0], [-1.5, 0]]
    vmax = [[3.5, 2.8], [3.5, 28]]
    #cmap = [[type_cmap, 'Reds'], [type_cmap, 'Blues']]
    cmap = [['Greys', 'Greys'], ['Greys', 'Greys']]

    fig = common.figure(figsize=(5.5, 4), box=debug)
    ax = fig.subplots(
        2, 2, sharex=True, sharey=True,
        #gridspec_kw=dict(width_ratios=(1,1,0.15)),
    )
    #ax[0, 2].set_visible(False)
    #ax[1, 2].set_visible(False)
    #ax[0, 2] = fig.add_axes([0.93, 0.1, 0.02, 0.4])
    #ax[1, 2] = fig.add_axes([0.93, 0.57, 0.02, 0.4])
    vticks = [0, 1, 5, 10, 50]
    xticks = [0.1, 0.5, 1, 5, 10, 50]

    label = [['a', 'c'], ['b', 'd']]
    pos = [[[-0.15, 0.8], [-0.1, 0.8]],
           [[-0.15, 0.8], [-0.1, 0.8]]]

    for i in range(2):
        for j in range(2):
            hm = df[name[i][j]].unstack(0)
            if j == 0:
                args = dict(cbar=False)
            else:
                args = dict(cbar=True)
            sns.heatmap(hm, vmin=vmin[i][j], vmax=vmax[i][j], cmap=cmap[i][j], ax=ax[i, j], **args)
            #sns.heatmap(hm, vmin=vmin[i][j], vmax=vmax[i][j], cmap=cmap[i][j], ax=ax[i, j])
            v = np.linspace(0, thr_v, 100)
            x = thr_x(v)
            v = v_loc(v)
            x = x_loc(x)
            ax[i, j].plot(v, x, c='k')
            ax[i, j].plot([v_loc(thr_v), v_loc(thr_v)], [x_loc(0.1), x_loc(10**2.1)], c='k')
            ax[i, j].invert_yaxis()
            ax[i, j].text(*pos[i][j], label[i][j], transform=ax[i, j].transAxes)
            ax[i, j].text(v_loc(5), x_loc(0.5), 'S')
            ax[i, j].text(v_loc(2), x_loc(6), 'C')
            ax[i, j].text(v_loc(50), x_loc(1), 'A')
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
    #cbar = ax[0, 0].collections[0].colorbar
    #cbar.set_ticks([0, 10, 20])
    #cbar.set_ticklabels([f'${{{l}}}$' for l in [0, 10, 20]])

    fig.savefig('figs/fig4.pdf')


if __name__ == '__main__':
    main(debug=False)
