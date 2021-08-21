import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from . import common


def f0(t, v, x):
    return np.cosh(v * t) * np.exp(-x * np.cosh(t))


def main(debug=False):
    fig = common.figure(figsize=(5.5, 4), box=debug)
    ax = fig.subplots(2, 2, sharex=True, sharey=True)

    t = np.linspace(0, 5, 1001)
    for v, x in [(0.5, 0.25), (0.5, 0.5), (1, 1), (1, 2), (2, 4)]:
        ax[0,0].plot(t, f0(t, v, x), label=f'$({v},{x})$')
    for v, x in [(1, 0.5), (1, 0.25), (2, 2), (2, 1), (3, 5), (3, 2)]:
        ax[0,1].plot(t, f0(t, v, x), label=f'$({v},{x})$')
    for v, x in [(0.1, 0.01), (0.1, 0.05), (0.1, 0.1), (1, 1), (1, 2), (1, 4), (1, 8)]:
        fvx = f0(t, v, x)
        fvx /= fvx.max()
        ax[1,0].plot(t, fvx, label=f'$({v},{x})$')
    for v, x in [(2,1), (2,2), (2,3), (20, 1), (20, 2), (20, 3)]:
        fvx = f0(t, v, x)
        fvx /= fvx.max()
        ax[1,1].plot(t, fvx, label=f'$({v},{x})$')

    ax[0,0].set_ylabel('$f(t)$')
    ax[1,0].set_ylabel('normalized $f(t)$')

    for i in range(2):
        for j in range(2):
            if i == 1:
                ax[i,j].set_xlabel('$t$')
            ax[i,j].legend()
            ax[i,j].xaxis.set_major_formatter('${:.2f}$'.format)
            ax[i,j].yaxis.set_major_formatter('${:.2f}$'.format)

    fig.savefig('figs/fig1.pdf')


if __name__ == '__main__':
    main()
