# Equations taken from https://www.dynamicmath.xyz/strange-attractors/

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Callable
from scipy.integrate import solve_ivp


def check_type(x, dim):
    if len(x) != dim:
        raise ValueError("Wrong dimension!")

    if isinstance(x, (list, np.ndarray)):
        return True
    else:
        raise TypeError("Check the input type of xyz, smh...")


def plot_3d(func: Callable, y_init: Union[tuple, list], arguments: tuple, title: str):
    solution = solve_ivp(fun=func,
                         t_span=(0, 100),
                         t_eval=np.linspace(0, 100, 50_000),
                         y0=y_init,
                         method="DOP853",
                         dense_output=True,
                         args=arguments)

    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*solution.y, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)

    plt.show()


def langford(t, xyz: Union[list, np.ndarray], a: float, b: float, c: float, d: float, e: float, f: float):
    if check_type(xyz, 3):
        x, y, z = xyz

    dxdt = (z - b) * x - d * y
    dydt = (d * x) + (z - b) * y
    dzdt = c + a * z - z **3 / 3 - (x ** 2 + y ** 2) * (1 + e * z) + f * z * x ** 3

    return np.array([dxdt, dydt, dzdt])


def halvorsen(t, xyz, a):
    if check_type(xyz, 3):
        x, y, z = xyz

    dxdt = - a * x - 4 * y - 4 * z - y ** 2
    dydt = - a * y - 4 * z - 4 * x - z ** 2
    dzdt = - a * z - 4 * x - 4 * y - x ** 2

    return np.array([dxdt, dydt, dzdt])


def rossler(t, xyz, a, b, c):
    if check_type(xyz, 3):
        x, y, z = xyz

    dxdt = - y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)

    return np.array([dxdt, dydt, dzdt])


if __name__ == "__main__":
    plot_3d(rossler, (.5, .8, .36), (0.2, 0.2, 5.7), "Rössler")
