import matplotlib.pyplot as plt


def figure(*args, box=False, **kwargs):
    plt.style.use('grayscale')
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    #plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams["font.size"] = 9
    plt.rcParams["mathtext.fontset"] = 'cm'
    plt.rcParams['mathtext.default'] = 'it'
    fig = plt.figure(*args, constrained_layout=True, **kwargs)
    if box:
        ax = fig.add_axes([0,0,1,1])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    return fig
