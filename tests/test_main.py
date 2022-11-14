import numpy as np

import exact as exact
import fem_front as femf


def test_sdr_1D():

    # Reference input:
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
        "n": 5
    }
    core_params = {
        "D":        1,
        "R":        0.8
    }
    source_params = {
        "function": "periodic",
        "alpha":    0.5,
        "beta" :    2,
        "gamma":    30
    }

    u_exact, x_exact = exact.ExactSolution().get_solution(pde, bc, bc_params, grid_params, core_params, source_params)
    u_fem, x_fem = femf.NumericalSolution().get_solution(pde, bc, bc_params, grid_params, core_params, source_params)
    
    assert np.linalg.norm(u_fem-u_exact) < 10**-2

