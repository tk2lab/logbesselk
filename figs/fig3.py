import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from . import common


def main(debug=False):
    df = pd.read_csv('figs/logk_time_results.csv')
    df['time'] *= 1000
    name = [['I', 'A'], ['S', 'C']]
    pos = [[[0.1, 0.85], [0.85, 0.1]], [[0.1, 0.1], [0.1, 0.85]]]

    fig = common.figure(figsize=(5.5, 4), box=debug)
    ax = fig.subplots(
        2, 3, sharex=True, sharey=True,
        gridspec_kw=dict(width_ratios=(1,1,0.15)),
    )
    ax[0, 2].set_visible(False)
    ax[1, 2].set_visible(False)
    cbar = fig.add_axes([0.93, 0.1, 0.02, 0.85])
    xticks = [0, 1, 5, 10, 50]
    yticks = [0.1, 0.5, 1, 5, 10, 50]

    for i in range(2):
        for j in range(2):
            hm = df.query(f'type=="{name[i][j]}"').pivot("x", "v", "time")
            if i == j == 0:
                args = dict(cbar_ax=cbar)
            else:
                args = dict(cbar=False)
            sns.heatmap(hm, vmin=0, vmax=28, cmap='Reds', ax=ax[i, j], **args)
            ax[i, j].invert_yaxis()
            ax[i, j].text(*pos[i][j], name[i][j], transform=ax[i, j].transAxes)
            ax[i, j].set_xticks([40*np.log10(x+1) for x in xticks])
            ax[i, j].set_xticklabels([f"${k}$" for k in xticks])
            ax[i, j].xaxis.set_ticks_position('both')
            ax[i, j].set_yticks([40*(np.log10(x)+1) for x in yticks])
            ax[i, j].set_yticklabels([f"${k}$" for k in yticks])
            ax[i, j].yaxis.set_ticks_position('both')
            if i == 1:
                ax[i, j].set_xlabel('$v$')
            else:
                ax[i, j].set_xlabel('')
            if j == 0:
                ax[i, j].set_ylabel('$x$')
            else:
                ax[i, j].set_ylabel('')
    cbar = ax[0, 0].collections[0].colorbar
    cbar.set_ticks([0, 10, 20])
    cbar.set_ticklabels([f'${{{l}}}\epsilon$' for l in [0, 10, 20]])

    fig.savefig('figs/fig3.pdf')


if __name__ == '__main__':
    main(debug=True)
