import numpy as np

import flexible_fem as fem


def test_laplace_1D():
    pde = "laplace_1D"
    bc_types = {
        "left": "neumann",
        "right": "dirichlet"
    }
    bc_params = {
        "left": ["constant", -1],
        "right": ["constant", 2]
        }
    grid_params = {
        "L": 1.7,
        "n": 139
    }
    core_params = {
        "D": ["constant", 1.5]
    }
    source_params = {
        "f": ["constant", 0]
    }

    u_exact, x_exact = fem.exact.ExactSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                              core_params, source_params)
    u_fem, x_fem = fem.front.NumericalSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                              core_params, source_params)

    assert np.square(u_fem-u_exact).mean() < 10**-12


def test_sad_1D():
    pde = "steady_advection_diffusion_1D"
    bc_types = {
        "left": "dirichlet",
        "right": "dirichlet"
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
        "A": ["constant", 1],
        "D": ["constant", 0.01],
        "R": ["linear", 0]
    }
    source_params = {
        "f": ["periodic", 0, 0, 20]
    }

    u_exact, x_exact = fem.exact.ExactSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                              core_params, source_params)
    u_fem, x_fem = fem.front.NumericalSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                              core_params, source_params)

    assert np.square(u_fem-u_exact).mean() < 10**-4


def test_sadr_1D():
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
        "R": ["linear", 1.3]
    }
    source_params = {
        "f": ["periodic", 0.8, 3.5, 30]
        }

    u_exact, x_exact = fem.exact.ExactSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                              core_params, source_params)
    u_fem, x_fem = fem.front.NumericalSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                              core_params, source_params)

    assert np.square(u_fem-u_exact).mean() < 10**-5


def test_sdr_1D_A():
    pde = "steady_diffusion_reaction_1D"
    bc_types = {
        "left": "neumann",
        "right": "neumann"
    }
    bc_params = {
        "left": ["constant", -1.5],
        "right": ["constant", -0.5]
    }
    grid_params = {
        "L": 1,
        "n": 100
    }
    core_params = {
        "D": ["constant", 1],
        "R": ["linear", 0.8]
    }
    source_params = {
        "f": ["periodic", 2.5, 2, 5]
    }

    u_exact, x_exact = fem.exact.ExactSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                              core_params, source_params)
    u_fem, x_fem = fem.front.NumericalSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                              core_params, source_params)

    assert np.square(u_fem-u_exact).mean() < 10**-9


def test_sdr_1D_B():
    pde = "steady_diffusion_reaction_1D"
    bc_types = {
        "left": "neumann",
        "right": "neumann"
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
        "D": ["constant", 1],
        "R": ["linear", 1]
    }
    source_params = {
        "f": ["periodic", 0, 1, 20]
    }

    u_exact, x_exact = fem.exact.ExactSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                              core_params, source_params)
    u_fem, x_fem = fem.front.NumericalSolution().get_solution(pde, bc_types, bc_params, grid_params,
                                                              core_params, source_params)

    assert np.square(u_fem-u_exact).mean() < 10**-12


# For debugging purposes
if __name__ == '__main__':
    test_sdr_1D_A()
