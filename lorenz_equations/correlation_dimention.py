import random
import numpy as np
from typing import Union
from matplotlib import pyplot as plt

import lorenz_attractor


def naive_correlation_dimension(solution_y: Union[list, np.ndarray], num_substeps: int, num_pts: int, slice: int):
    solution_y = solution_y.T       # to get the points (x, y, z)
    epsilons = np.linspace(1, 10, num_substeps)       # set the different values of neighborhood
    N_array = np.full(shape=(num_pts, num_substeps), fill_value=(-1))       # array for the number of point in the neighborhood of a point x, depending on ε

    for i in range(num_pts):
        n = random.randint(slice+1, len(solution_y[:, 0]-1))        # choose points to use for calculation

        pt = solution_y[n, :]
        for e in range(num_substeps):
            for point in solution_y:
                dist = np.linalg.norm(point - pt)       # float
                if dist <= epsilons[e]:
                    N_array[i, e] += 1

    return epsilons, N_array


eps, N = naive_correlation_dimension(lorenz_attractor.solution.y, 10, 250, 1000)
C = np.average(N, axis=0)
log_eps = np.log(eps)
log_C = np.log(C)

more_eps_pts = np.linspace(log_eps[0], log_eps[-1])
model_params, cov_matrix = np.polyfit(log_eps, log_C, 1, cov=True)
model_data = np.polyval(model_params, more_eps_pts)

fig, ax2 = plt.subplots(1, 1, figsize=(4, 4))
ax2.set_xlabel("log(ε)")
ax2.set_ylabel(r"log(C(ϵ))")
ax2.scatter(log_eps, log_C, color="black")
ax2.plot(more_eps_pts, model_data, color="red", label=f"d={model_data[0]:.2f}")
ax2.legend(loc="best")
plt.tight_layout()
plt.show()
