# steady advection diffusion 1D

import matplotlib.pyplot as plt
import flexible_fem as fem

pde = "steady_advection_diffusion_1D"
bc = {
    "left": ["dirichlet", 0],
    "right": ["dirichlet", 1]
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
    "A":        1,
    "D":        0.01,
    "R":        0
}
source_params = {
    "function": "periodic",
    "alpha":    0,
    "beta":     0,
    "gamma":    20
}

u_exact, x_exact = fem.exact.ExactSolution().get_solution(pde, bc, bc_params, grid_params,
                                                          core_params, source_params)
u_fem, x_fem = fem.front.NumericalSolution().get_solution(pde, bc, bc_params, grid_params,
                                                          core_params, source_params)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x_fem, u_fem, linewidth=5, label='fem')
ax.plot(x_exact, u_exact, linestyle=':', linewidth=5, label='exact')

title = pde
ax.set_title(title, y=1.05, fontsize=14)
ax.legend()
ax.grid()
plt.show()
