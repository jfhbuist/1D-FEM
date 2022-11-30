# laplace 1D

import matplotlib.pyplot as plt
import numpy as np
import flexible_fem as fem

pde = "laplace_2D"
bc = {
    "left": ["dirichlet"],
    "right": ["dirichlet"],
    "upper": ["dirichlet"],
    "lower": ["dirichlet"]
}
grid_params = {
    "L": 1,
    "H": 1,
    "nx": 100,
    "ny": 100
}
bc_params = {
    "left": ["sine", 0, 1, np.pi/(grid_params["H"]), 0, 0],
    "right": ["sine", 0, 1, np.pi/(grid_params["H"]), 0, 0],
    "upper": ["sine", 0, 0, 0, 0, 0],
    "lower": ["sine", 0, 0, 0, 0, 0]
}
core_params = {
    "D":        1
}
source_params = {
    "function": "zero",
}

u_exact, x_exact, y_exact = fem.exact.ExactSolution().get_solution(pde, bc, bc_params, grid_params,
                                                                   core_params, source_params)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x_exact, y_exact, u_exact)  # , label = 'exact')

L = grid_params["L"]
H = grid_params["H"]
nx = grid_params["nx"]
ny = grid_params["ny"]
ax.set_xlim(0, L)
ax.set_ylim(0, H)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
title = pde
ax.set_title(title, y=1.05, fontsize=14)
plt.show()
