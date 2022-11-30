# steady diffusion reaction 1D

import matplotlib.pyplot as plt
import flexible_fem as fem

# # Reference input:
# pde = "steady_diffusion_reaction_1D"
# bc = {
#     "left": ["neumann", 0],
#     "right": ["neumann", 0]
# } 
# bc_params = {
#     "left": ["constant", 0],
#     "right": ["constant", 0]
# }
# grid_params = {
#     "L": 1,
#     "n": 5
# }
# core_params = {
#     "D":        1,
#     "R":        0.8
# }
# source_params = {
#     "function": "periodic",
#     "alpha":    0.5,
#     "beta":     2,
#     "gamma":    30
# }

# # set 12
# pde = "steady_diffusion_reaction_1D"
# bc = {
#     "left": ["neumann", 0],
#     "right": ["neumann", 0]
# } 
# bc_params = {
#     "left": ["constant", 0],
#     "right": ["constant", 0]
# }
# grid_params = {
#     "L": 1,
#     "n": 100
# }
# core_params = {
#     "D":        1,
#     "R":        1
# }
# source_params = {
#     "function": "periodic",
#     "alpha":    1,
#     "beta":     0,
#     "gamma":    0,
# }

# set 13
pde = "steady_diffusion_reaction_1D"
bc = {
    "left": ["neumann", 0],
    "right": ["neumann", 0]
} 
bc_params = {
    "left": ["constant", 0],
    "right": ["constant", 0]
}
grid_params = {
    "L": 1,
    "n": 100
}
core_params = {
    "D":        1,
    "R":        1
}
source_params = {
    "function": "periodic",
    "alpha":    0,
    "beta":     1,
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
