import numpy as np

import flexible_fem as fem


def test_laplace_2D_A():
    pde = "laplace_2D"
    bc_types = {
        "left": "dirichlet",
        "right": "dirichlet",
        "bottom": "dirichlet",
        "top": "dirichlet",
    }
    grid_params = {
        "L": 1,
        "H": 1,
        "nx": 4,
        "ny": 6
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
    u_fem, x_fem, y_fem = fem.front.NumericalSolution().get_solution(pde, bc_types, bc_params,
                                                        grid_params, core_params, source_params)

    assert np.square(u_fem-u_exact).mean() < 10**-3


# For debugging purposes
if __name__ == '__main__':
    test_laplace_2D_A()
