import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from scipy.integrate import solve_ivp


def lorenz_equations(t, xyz: Union[list, np.ndarray], sigma: float, r: float, b: float):
    if len(xyz) != 3:
        raise ValueError("Wrong dimension!")

    if isinstance(xyz, (list, np.ndarray)):
        x, y, z = xyz
    else:
        raise TypeError("Check the input type of xyz, smh...")

    dxdt = sigma * (y - x)
    dydt = r * x - y - x * z
    dzdt = x * y - b * z
    return np.array([dxdt, dydt, dzdt])


solution = solve_ivp(fun=lorenz_equations,
                     t_span=(0, 3700),
                     t_eval=np.linspace(0, 3700, 200_000),
                     y0=[0., 1., 0.],
                     method="DOP853",
                     dense_output=True,
                     args=(10, 28, 8/3))

#ax = plt.figure().add_subplot(projection='3d')

#ax.plot(*solution.y, lw=0.3)
#ax.set_xlabel("X Axis")
#ax.set_ylabel("Y Axis")
#ax.set_zlabel("Z Axis")
#plt.show()

#fig, ax1 = plt.subplots(1, 1, figsize=(4, 4))
#ax1.set_xlabel("x")
#ax1.set_ylabel(r"z")
#ax1.plot(solution.y[0], solution.y[2], label=f"(σ, r, b)=(10, 28, 8/3)", lw=0.5)
#ax1.legend(loc="best")
#plt.tight_layout()
#plt.show()
