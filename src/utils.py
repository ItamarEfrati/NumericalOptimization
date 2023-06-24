import numpy as np
from matplotlib import pyplot as plt

from tests.examples import example_3, rosenbrock_function, example_5, example_6

style = ['r--', 'b--', 'g--', 'm--']


def plot_example(methods, example, name):
    levels = 60
    start = -3
    stop = 3
    if name in ['example_6', 'rosenback']:
        levels = [0.01, 0.05, 0.1, 0.2, 0.3, 1, 5, 10, 50, 150, 300, 500]
    elif name == 'example_5':
        start = -400
        stop = 400

    xlist = np.linspace(start, stop, 100)
    ylist = np.linspace(start, stop, 100)
    X, Y = np.meshgrid(xlist, ylist)
    X, Y = X.reshape(-1), Y.reshape(-1)

    Z = np.array(list(map(lambda i: example(np.array([X[i], Y[i]]), False)[0], range(100 * 100))))

    X, Y, Z = X.reshape(100, 100), Y.reshape(100, 100), Z.reshape(100, 100)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    ax = axs[0]
    cp = ax.contour(X, Y, Z, levels=levels)
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title(name)
    for i, (method, optimizer) in enumerate(methods.items()):
        path = np.array(optimizer.locations)
        ax.plot(path[:, 0], path[:, 1], style[i], label=method)
    ax.legend()
    ax = axs[1]
    for i, (method, optimizer) in enumerate(methods.items()):
        ax.plot(optimizer.f_values, label=method)
    ax.set_title("Values")
    ax.legend()
    plt.show()
