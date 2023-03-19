# steady diffusion reaction 2D

import matplotlib.pyplot as plt
# import numpy as np
import flexible_fem as fem

pde = "steady_diffusion_reaction_2D"
bc_types = {
    "left": "dirichlet",
    "right": "dirichlet",
    "bottom": "dirichlet",
    "top": "dirichlet",
}
grid_params = {
    "L": 1,
    "H": 1,
    "nx": 30,
    "ny": 30
}
bc_params = {
    "left": ["constant", 0],
    "right": ["constant", 1],
    "bottom": ["constant", 0],
    "top": ["constant", 1],
}
core_params = {
    "D": ["constant", -1],
    "R": ["exponential", 3]
}
source_params = {
    "f": ["constant", 0]
}

u_fem, x_fem, y_fem = fem.front.NumericalSolution().get_solution(pde, bc_types, bc_params,
                                                        grid_params, core_params, source_params)

L = grid_params["L"]
H = grid_params["H"]
nx = grid_params["nx"]
ny = grid_params["ny"]
title1 = "thermal explosion"

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.plot_surface(x_fem, y_fem, u_fem, label='numerical')

ax1.set_xlim(0, L)
ax1.set_ylim(0, H)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')
ax1.set_title(title1, y=1.05, fontsize=14)

plt.show()
