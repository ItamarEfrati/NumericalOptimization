import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def plot_qp_example(path, f_values, ineq_values):
    print("Quadratic")

    ineq_values, descriptions = list(zip(*ineq_values))
    f_values, f_p_values, t_values = list(zip(*f_values))

    f_values = list(map(lambda x, y: x / y, f_values, t_values))

    print('Final Candidate', path[-1])
    print('Final Candidate Objective Value', f_values[-1])
    print('Final Candidate Barrier Value', f_p_values[-1])

    for i in range(len(ineq_values)):
        print(f"Ineq {i + 1}, {descriptions[i]} value is {ineq_values[i]}")

    print(f"Eq x + y + z = 1, value is {np.sum(path[-1])}")

    fig, ax = plt.subplots()
    ax.plot(f_values)
    ax.set_title("Quadratic Original Objective")
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Value")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(f_p_values)
    ax.set_title("Quadratic Barrier Objective With Increasing t Value")
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Value")
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    path = np.squeeze(np.array(path))
    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color='lightpink', alpha=0.5)
    ax.view_init(elev=45, azim=45)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], style[1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def plot_lp_example(path, f_values, ineq_values):
    print("Linear")

    ineq_values, descriptions = list(zip(*ineq_values))
    f_values, f_p_values, t_values = list(zip(*f_values))

    f_values = list(map(lambda x, y: x / y, f_values, t_values))

    for i in range(len(ineq_values)):
        print(f"Ineq {i + 1}, {descriptions[i]} value is {ineq_values[i]}")

    print('Final Candidate', path[-1])
    print('Final Candidate Objective Value', f_values[-1])
    print('Final Candidate Barrier Value', f_p_values[-1])

    fig, ax = plt.subplots()
    ax.plot(f_values)
    ax.set_title("Linear Original Objective")
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Value")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(f_p_values)
    ax.set_title("Linear Barrier Objective With Increasing t Value")
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Value")
    plt.show()

    fig, ax = plt.subplots()
    path = np.squeeze(np.array(path))
    points = np.array([[2, 1], [2, 0], [1, 0], [0, 1]])
    ax.fill(points[:, 0], points[:, 1], color='lightpink', alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], style[1])
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 2)
    ax.grid()
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.show()
