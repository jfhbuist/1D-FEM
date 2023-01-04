# laplace 1D

import matplotlib.pyplot as plt
import numpy as np
import flexible_fem as fem

pde = "laplace_1D"
# neumann bc: set value of gradient of solution normal to boundary
# dirichlet bc: set value of solution at boundary
bc = {
    "left": ["neumann", -1],
    "right": ["dirichlet", 2]
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
    "D":        1.5,
}
source_params = {
    "function": "zero"
    }

u_exact, x_exact = fem.exact.ExactSolution().get_solution(pde, bc, bc_params, grid_params,
                                                          core_params, source_params)
u_fem, x_fem = fem.front.NumericalSolution().get_solution(pde, bc, bc_params, grid_params,
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
