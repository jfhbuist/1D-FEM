# steady advection diffusion reaction 1D

import matplotlib.pyplot as plt
import numpy as np
import flexible_fem as fem

# test all parameters
pde = "steady_advection_diffusion_reaction_1D"
bc_types = {
    "left": "dirichlet",
    "right": "neumann"
}
bc_params = {
    "left": ["constant", 1],
    "right": ["constant", 0]
    }
grid_params = {
    "L": 1.7,
    "n": 139
}
core_params = {
    "A": ["constant", 0.5],
    "D": ["constant", 0.01],
    "R": ["constant", 1.3]
}
source_params = {
    "f": ["periodic", 0.8, 3.5, 30]
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
