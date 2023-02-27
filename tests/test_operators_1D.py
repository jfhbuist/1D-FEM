import numpy as np

import flexible_fem as fem


def test_source_1D():
    # Test the construction of the source operator

    reference_source = np.array([0.12082889, 0.13589522, 0.13255334, 0.11934129, 0.04776451])

    L = 1
    n = 5
    alpha = 0.5
    beta = 2
    gamma = 30

    # periodic source term:
    f = lambda xy: alpha + beta*np.sin(gamma*xy[0])

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization(dim)

    source = fem.core.Source(grid, discretization, f)
    # print(source.d)

    assert np.square(source.d-reference_source).max() < 10**-8


def test_diffusion_1D():
    # Test the construction of the diffusion operator

    reference_stiffness = np.array([
                                [ 5.2,  -5.2,   0.0,   0.0,   0.0],
                                [-5.2,  10.4,  -5.2,   0.0,   0.0],
                                [ 0.0,  -5.2,  10.4,  -5.2,   0.0],
                                [ 0.0,   0.0,  -5.2,  10.4,  -5.2],
                                [ 0.0,   0.0,   0.0,  -5.2,   5.2]
                                ])

    D = 1.3
    L = 1
    n = 5
    bc_types = {
        "left": "neumann",
        "right": "neumann"
    }
    bc_params = {
        "left": ["constant", 0],
        "right": ["constant", 0]
    }

    bc_functions = {}
    for lb in bc_params:
        if bc_params[lb][0] == "constant":
            bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization(dim)

    diffusion = fem.core.Diffusion(grid, discretization, bc_types, bc_functions, D)

    operators = [diffusion]
    stiffness = fem.core.SolutionOperator(grid, discretization, operators)
    # print(stiffness.s)

    assert np.square(stiffness.s-reference_stiffness).max() < 10**-8


def test_reaction_1D():
    # Test the construction of the reaction operator

    reference_stiffness = np.array([
                                [0.06666667,  0.03333333,  0.0,         0.0,         0.0       ],
                                [0.03333333,  0.13333333,  0.03333333,  0.0,         0.0       ],
                                [0.0,         0.03333333,  0.13333333,  0.03333333,  0.0       ],
                                [0.0,         0.0,         0.03333333,  0.13333333,  0.03333333],
                                [0.0,         0.0,         0.0,         0.03333333,  0.06666667]
                                ])

    R = 0.8
    L = 1
    n = 5
    bc_types = {
        "left": "neumann",
        "right": "neumann"
    }
    bc_params = {
        "left": ["constant", 0],
        "right": ["constant", 0]
    }
    bc_functions = {}
    for lb in bc_params:
        if bc_params[lb][0] == "constant":
            bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization(dim)

    reaction = fem.core.Reaction(grid, discretization, bc_types, bc_functions, R)

    operators = [reaction]
    stiffness = fem.core.SolutionOperator(grid, discretization, operators)
    # print(stiffness.s)

    assert np.square(stiffness.s-reference_stiffness).max() < 10**-8


def test_advection_1D():
    # Test the construction of the advection operator

    reference_stiffness = np.array([
                                    [-0.45,  0.45,  0.0,   0.0,   0.0 ],
                                    [-0.45,  0.0,   0.45,  0.0,   0.0 ],
                                    [ 0.0,  -0.45,  0.0,   0.45,  0.0 ],
                                    [ 0.0,   0.0,  -0.45,  0.0,   0.45],
                                    [ 0.0,   0.0,   0.0,  -0.45,  0.45]
                                    ])

    A = 0.9
    L = 1
    n = 5
    bc_types = {
        "left": "neumann",
        "right": "neumann"
    }
    bc_params = {
        "left": ["constant", 0],
        "right": ["constant", 0]
    }
    bc_functions = {}
    for lb in bc_params:
        if bc_params[lb][0] == "constant":
            bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization(dim)

    advection = fem.core.Advection(grid, discretization, bc_types, bc_functions, A)

    operators = [advection]
    stiffness = fem.core.SolutionOperator(grid, discretization, operators)
    # print(stiffness.s)

    assert np.square(stiffness.s-reference_stiffness).max() < 10**-8


def test_natural_boundary_1D():
    # Test the construction of the natural boundary vector for a diffusion operator

    reference = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5])

    # Laplace equation:
    # - D*u_xx = 0
    D = 1.5
    L = 1.7
    n = 7
    bc_types = {
        "left": "dirichlet",
        "right": "neumann"
    }
    bc_params = {
        "left": ["constant", 2],
        "right": ["constant", -1]
    }
    bc_functions = {}
    for lb in bc_params:
        if bc_params[lb][0] == "constant":
            bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization(dim)

    diffusion = fem.core.Diffusion(grid, discretization, bc_types, bc_functions, D)

    operators = [diffusion]

    natural_boundary = fem.core.NaturalBoundary(grid, discretization, bc_types, bc_functions, operators)

    assert np.square(natural_boundary.b_nat-reference).max() < 10**-8


def test_solution_1D():
    # Test all the steps up to and including the obtainment of the finite element solution

    reference_solution = np.array([0.71743664, 0.70506221, 0.69397784, 0.68446714, 0.67938135])

    # Diffusion-reaction equation (aka Helmholtz equation):
    # -D*u_xx + R*u = f
    D = 1
    R = 0.8
    L = 1
    n = 5
    alpha = 0.5
    beta = 2
    gamma = 30
    bc_types = {
        "left": "neumann",
        "right": "neumann"
    }
    bc_params = {
        "left": ["constant", 0],
        "right": ["constant", 0]
    }
    bc_functions = {}
    for lb in bc_params:
        if bc_params[lb][0] == "constant":
            bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]

    # periodic source term:
    f = lambda xy: alpha + beta*np.sin(gamma*xy[0])

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization(dim)

    source = fem.core.Source(grid, discretization, f)

    diffusion = fem.core.Diffusion(grid, discretization, bc_types, bc_functions, D)

    reaction = fem.core.Reaction(grid, discretization, bc_types, bc_functions, R)

    operators = [diffusion, reaction]
    stiffness = fem.core.SolutionOperator(grid, discretization, operators)
    # print(stiffness.s)

    natural_boundary = fem.core.NaturalBoundary(grid, discretization, bc_types, bc_functions, operators)

    # specify points at which to return function:
    x_vec = np.linspace(0, L, n)
    xy = [[x] for x in x_vec]

    solution = fem.core.Solution(grid, discretization, bc_types, bc_functions, stiffness, source, natural_boundary, xy)

    assert np.square(solution.u-reference_solution).max() < 10**-8


def test_solution_interpolation_1D():
    # Test the capability to return the solution at points other than the element vertices

    reference_solution = np.array([0.71808674, 0.70630158, 0.69881718, 0.69456684, 0.68971147,
                                   0.68407707, 0.67982015])

    # Diffusion-reaction equation (aka Helmholtz equation):
    # -D*u_xx + R*u = f
    D = 1
    R = 0.8
    L = 1
    n = 23
    alpha = 0.5
    beta = 2
    gamma = 30
    bc_types = {
        "left": "neumann",
        "right": "neumann"
    }
    bc_params = {
        "left": ["constant", 0],
        "right": ["constant", 0]
    }
    bc_functions = {}
    for lb in bc_params:
        if bc_params[lb][0] == "constant":
            bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]

    # periodic source term:
    f = lambda xy: alpha + beta*np.sin(gamma*xy[0])

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization(dim)

    source = fem.core.Source(grid, discretization, f)

    diffusion = fem.core.Diffusion(grid, discretization, bc_types, bc_functions, D)

    reaction = fem.core.Reaction(grid, discretization, bc_types, bc_functions, R)

    operators = [diffusion, reaction]
    stiffness = fem.core.SolutionOperator(grid, discretization, operators)
    # print(stiffness.s)

    natural_boundary = fem.core.NaturalBoundary(grid, discretization, bc_types, bc_functions, operators)

    # specify points at which to return function:
    x_vec = np.linspace(0, L, int(np.floor(n/3)))
    xy = [[x] for x in x_vec]

    solution = fem.core.Solution(grid, discretization, bc_types, bc_functions, stiffness, source, natural_boundary, xy)

    assert np.square(solution.u-reference_solution).max() < 10**-8


# For debugging purposes
if __name__ == '__main__':
    test_solution_1D()
