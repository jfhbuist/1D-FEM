# laplace 2D

import matplotlib.pyplot as plt
import numpy as np
import flexible_fem as fem

pde = "laplace_2D"
bc_types = {
    "left": "neumann",
    "right": "dirichlet",
    "bottom": "dirichlet",
    "top": "dirichlet",
}
# grid_params = {
#     "L": 1,
#     "H": 1,
#     "nx": 4,
#     "ny": 6
# }
grid_params = {
    "L": 1,
    "H": 1,
    "nx": 20,
    "ny": 30
}
bc_params = {
    "left": ["sine", 0, 1, np.pi/(grid_params["H"]), 0, 0],  # g(y) = a + b*sin(c*y)
    "right": ["sine", 0, 1, np.pi/(grid_params["H"]), 0, 0],  # g(y) = a + b*sin(c*y)
    "bottom": ["sine", 0, 0, 0, 0, 0],  # g(x) = a + b*sin(c*x)
    "top": ["sine", 0, 0, 0, 0, 0],  # g(x) = a + b*sin(c*x)
}
core_params = {
    "D":        1
}
source_params = {
    "function": "zero",
}

u_exact, x_exact, y_exact = fem.exact.ExactSolution().get_solution(pde, bc_types, bc_params,
                                                        grid_params, core_params, source_params)
# u_fem, x_fem, y_fem = fem.front.NumericalSolution().get_solution(pde, bc_types, bc_params,
#                                                        grid_params, core_params, source_params)

# print("MSE = {0:.2e}".format(np.square(u_fem-u_exact).mean()))

L = grid_params["L"]
H = grid_params["H"]
nx = grid_params["nx"]
ny = grid_params["ny"]
title1 = pde + " exact"
title2 = pde + " numerical"

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.plot_surface(x_exact, y_exact, u_exact, label='exact')
ax1.set_xlim(0, L)
ax1.set_ylim(0, H)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')
ax1.set_title(title1, y=1.05, fontsize=14)

# fig2 = plt.figure()
# ax2 = fig2.add_subplot(projection='3d')
# ax2.plot_surface(x_fem, y_fem, u_fem, label='numerical')

# ax2.set_xlim(0, L)
# ax2.set_ylim(0, H)
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_zlabel('u')
# ax2.set_title(title2, y=1.05, fontsize=14)

plt.show()
