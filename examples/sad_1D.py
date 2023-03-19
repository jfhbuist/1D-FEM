# steady advection diffusion 1D

import matplotlib.pyplot as plt
import numpy as np
import flexible_fem as fem

pde = "steady_advection_diffusion_1D"
bc_types = {
    "left": "dirichlet",
    "right": "dirichlet"
}
bc_params = {
    "left": ["constant", 0],
    "right": ["constant", 1]
}
grid_params = {
    "L": 1,
    "n": 100
}
core_params = {
    "A": ["constant", 1],
    "D": ["constant", 0.01],
    "R": ["linear", 0]
}
source_params = {
    "f": ["periodic", 0, 0, 20]
}

u_exact, x_exact = fem.exact.ExactSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                          core_params, source_params)
u_fem, x_fem = fem.front.NumericalSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                          core_params, source_params)

print("MSE = {0:.2e}".format(np.square(u_fem-u_exact).mean()))

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x_fem, u_fem, linewidth=5, label='fem')
ax.plot(x_exact, u_exact, linestyle=':', linewidth=5, label='exact')

title = pde
ax.set_title(title, y=1.05, fontsize=14)
ax.legend()
ax.grid()
plt.show()
