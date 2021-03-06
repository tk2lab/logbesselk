import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from . import common


def f0(t, v, x):
    return np.cosh(v * t) * np.exp(-x * np.cosh(t))


def main(debug=False):
    fig = common.figure(figsize=(5.5, 4), box=debug)
    ax = fig.subplots(2, 2, sharex=True, sharey='row')

    t = np.linspace(0, 5, 1001)
    for i, (v, x) in enumerate([(0.5, 0.25), (0.5, 0.5), (1, 1), (1, 2), (2, 4)]):
        if i < 2:
            ax[0,0].plot(t, f0(t, v, x), label=f'$({v},{x})$', ls='-.')
        elif i < 4:
            ax[0,0].plot(t, f0(t, v, x), label=f'$({v},{x})$', ls='--')
        else:
            ax[0,0].plot(t, f0(t, v, x), label=f'$({v},{x})$', ls='-')
    for i, (v, x) in enumerate([(1, 0.5), (1, 0.25), (2, 2), (2, 1), (3, 5), (3, 2)]):
        if i < 2:
            ax[0,1].plot(t, f0(t, v, x), label=f'$({v},{x})$', ls='-.')
        elif i < 4:
            ax[0,1].plot(t, f0(t, v, x), label=f'$({v},{x})$', ls='--')
        else:
            ax[0,1].plot(t, f0(t, v, x), label=f'$({v},{x})$')
    for i, (v, x) in enumerate([(0.1, 0.01), (0.1, 0.05), (0.1, 0.1), (1, 1), (1, 2), (1, 4), (1, 8)]):
        fvx = f0(t, v, x)
        fvx /= fvx.max()
        if i < 3:
            ax[1,0].plot(t, fvx, label=f'$({v},{x})$')
        elif i // 4 == 1:
            ax[1,0].plot(t, fvx, label=f'$({v},{x})$', ls='--')
    for i, (v, x) in enumerate([(2,1), (2,2), (2,3), (20, 1), (20, 2), (20, 3)]):
        fvx = f0(t, v, x)
        fvx /= fvx.max()
        if i < 3:
            ax[1,1].plot(t, fvx, label=f'$({v},{x})$')
        elif i // 4 == 1:
            ax[1,1].plot(t, fvx, label=f'$({v},{x})$', ls='--')

    ax[0,0].set_ylabel('$f(t)$')
    ax[1,0].set_ylabel('normalized $f(t)$')

    name = [['a', 'c'], ['b', 'd']]
    pos = [[[0.1, 0.85], [0.1, 0.85]], [[0.1, 0.15], [0.1, 0.15]]]
    for i in range(2):
        for j in range(2):
            ax[i, j].text(*pos[i][j], name[i][j], transform=ax[i, j].transAxes)
            if i == 1:
                ax[i,j].set_xlabel('$t$')
            ax[i,j].legend(loc='upper right')
            ax[i,j].xaxis.set_major_formatter('${:.2f}$'.format)
            ax[i,j].yaxis.set_major_formatter('${:.2f}$'.format)

    fig.savefig('figs/fig1.pdf')


if __name__ == '__main__':
    main()
