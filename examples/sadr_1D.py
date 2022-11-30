# steady advection diffusion reaction 1D

import matplotlib.pyplot as plt
import numpy as np
import flexible_fem as fem

# test all parameters
pde = "steady_advection_diffusion_reaction_1D"
bc = {
    "left": ["dirichlet", 1],
    "right": ["neumann", 0]
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
    "A":        0.5,
    "D":        0.01,
    "R":        1.3
}
source_params = {
    "function": "periodic",
    "alpha":    0.8,
    "beta":     3.5,
    "gamma":    30
    }

u_exact, x_exact = fem.exact.ExactSolution().get_solution(pde, bc, bc_params, grid_params,
                                                          core_params, source_params)
u_fem, x_fem = fem.front.NumericalSolution().get_solution(pde, bc, bc_params, grid_params,
                                                          core_params, source_params)

print(np.square(u_fem-u_exact).mean())

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x_fem, u_fem, linewidth=5, label='fem')
ax.plot(x_exact, u_exact, linestyle=':', linewidth=5, label='exact')

title = pde
ax.set_title(title, y=1.05, fontsize=14)
ax.legend()
ax.grid()
plt.show()
