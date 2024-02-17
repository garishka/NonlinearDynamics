import numpy as np
import matplotlib.pyplot as plt
from typing import Union

preamble = [r'\usepackage[utf8]{inputenc}',
    r'\usepackage[bulgarian]{babel}',
    r"\usepackage{amsmath}",
    r'\usepackage{siunitx}']
LaTeX = {
"text.usetex": True,
"font.family": "CMU Serif",
    "pgf.preamble": "\n".join(line for line in preamble),
    "pgf.rcfonts": True,
    "pgf.texsystem": "lualatex"}
plt.rcParams.update(LaTeX)


def logistic(r, x):
    return r * x * (1 - x)


def plot_logistic(x_interval: tuple, r: Union[list, float]):
    """ Plots y'(x)=rx(1-x) for x ∈ x_interval."""
    x = np.linspace(*x_interval)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_title("Logistic equation")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$y'=f_r(x)$")
    for el in r:
        ax.plot(x, logistic(el, x), label=f"r={el}")
    ax.legend(loc="best")
    plt.show()


def plot_bif_diag(x0: float, r_interval: tuple, r_density: int, num_iter: int, num_skip: int):
    """Plot the bifurcation diagram for the logistic equation.

    Parameters
    -----------
    x0: float
        Initial value of the x variable in the equation.
    r_interval: tuple
        Range of values for the parameter 'r'.
    r_density: int
        Number of 'r' values evenly spaced within the interval.
    num_iter: int
        Number of iterations for each 'r' value.
    num_skip: int
        Number of iterations to skip before recording results to ensure stability.

    Returns
    --------
    None
    """

    if num_skip >= num_iter:
        raise ValueError("num_skip must be less than num_iter")

    r_values = np.linspace(*r_interval, r_density)
    Y = np.zeros(shape=(r_density * num_iter,))
    R = np.zeros(shape=(r_density * num_iter,))
    id = 0
    plt.figure(figsize=(8, 4))

    for r in r_values:
        x = x0

        # запазваме само последните num_iter точки
        for i in range(num_iter + num_skip):
            if i >= num_skip:
                R[id] = r
                Y[id] = x
                id += 1

            # logistic map function (x_{n+1} = r x_n (1 - x_n)
            x = logistic(r, x)

        plt.scatter(R, Y, color="red", marker=".", s=0.2)

    plt.xlabel('r')
    plt.ylabel('x')
    plt.title('Bifurcation Diagram of the Logistic Map')
    plt.show()


# по някаква причина решава, че ще начертае (0, 0), не ми се занимава
if __name__ == "__main__":
    plot_bif_diag(x0=0.2, r_interval=(2.6, 4.), r_density=1_000, num_iter=200, num_skip=100)
