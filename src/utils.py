import numpy as np
from matplotlib import pyplot as plt

from tests.examples import example_3, rosenbrock_function

style = ['rs', 'b^', 'g--', 'm.']


def plot_example_1(methods):
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt(X ** 2 + Y ** 2)

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title(f'Example 1')
    for i, (method, optimizer) in enumerate(methods.items()):
        path = np.array(optimizer.locations)
        plt.plot(path[:, 0], path[:, 1], style[i], label=method)
    plt.legend()
    plt.legend()
    plt.show()


def plot_example_2(methods):
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt(X ** 2 + 100 * (Y ** 2))

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title(f'Example 2')

    for i, (method, optimizer) in enumerate(methods.items()):
        path = np.array(optimizer.locations)
        plt.plot(path[:, 0], path[:, 1], style[i], label=method)
    plt.legend()
    plt.legend()
    plt.show()


def plot_example_3(methods):
    xlist = np.linspace(-3, 3, 100)
    ylist = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(xlist, ylist)
    X, Y = X.reshape(-1), Y.reshape(-1)
    Z = np.array(list(map(lambda i: example_3(np.array([X[i], Y[i]]), False)[0], range(100 * 100))))

    X, Y, Z = X.reshape(100, 100), Y.reshape(100, 100), Z.reshape(100, 100)
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z, levels=[0.01, 0.05, 0.1, 0.2, 0.3, 1, 5, 10, 50, 150, 300, 500])
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title(f'Example 3')
    for i, (method, optimizer) in enumerate(methods.items()):
        path = np.array(optimizer.locations)
        plt.plot(path[:, 0], path[:, 1], style[i], label=method)
    plt.legend()
    plt.show()


def plot_rosenbrock_function(methods):
    xlist = np.linspace(-2.5, 2.5, 100)
    ylist = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(xlist, ylist)
    X, Y = X.reshape(-1), Y.reshape(-1)
    Z = np.array(list(map(lambda i: rosenbrock_function(np.array([X[i], Y[i]]), False)[0], range(100 * 100))))

    X, Y, Z = X.reshape(100, 100), Y.reshape(100, 100), Z.reshape(100, 100)
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z, levels=[0.01, 0.05, 0.1, 0.2, 0.3, 1, 5, 10, 50, 150, 300, 500])
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title(f'rosenbrock_function')
    for i, (method, optimizer) in enumerate(methods.items()):
        path = np.array(optimizer.locations)
        plt.plot(path[:, 0], path[:, 1], style[i], label=method)
    plt.legend()
    plt.show()
