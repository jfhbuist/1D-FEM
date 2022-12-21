import numpy as np

import flexible_fem as fem


def test_source_1D():

    reference_source = np.array([0.12082889, 0.13589522, 0.13255334, 0.11934129, 0.04776451])

    L = 1
    n = 5
    alpha = 0.5
    beta = 2
    gamma = 30

    # periodic source term:
    f = lambda x: alpha + beta*np.sin(gamma*x)

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization()

    source = fem.core.Source(grid, discretization, f)
    # print(source.d)

    assert np.square(source.d-reference_source).max() < 10**-8


def test_diffusion_1D():

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

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization()

    diffusion = fem.core.Diffusion(D)

    operators = [diffusion]
    stiffness = fem.core.StiffnessMatrix(grid, discretization, operators)
    # print(stiffness.s)

    assert np.square(stiffness.s-reference_stiffness).max() < 10**-8


def test_reaction_1D():

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

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization()

    reaction = fem.core.Reaction(R)

    operators = [reaction]
    stiffness = fem.core.StiffnessMatrix(grid, discretization, operators)
    # print(stiffness.s)

    assert np.square(stiffness.s-reference_stiffness).max() < 10**-8


def test_advection_1D():

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

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization()

    advection = fem.core.Advection(A)

    operators = [advection]
    stiffness = fem.core.StiffnessMatrix(grid, discretization, operators)
    # print(stiffness.s)

    assert np.square(stiffness.s-reference_stiffness).max() < 10**-8


def test_natural_boundary_1D():
    reference = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5])

    # Laplace equation:
    # - D*u_xx = 0
    D = 1.5
    L = 1.7
    n = 7

    bc = {
        "left": ["dirichlet", 2],
        "right": ["neumann", -1]
    }

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization()

    diffusion = fem.core.Diffusion(D)

    operators = [diffusion]

    natural_boundary = fem.core.NaturalBoundary(grid, discretization, operators, bc)

    assert np.square(natural_boundary.b_nat-reference).max() < 10**-8


def test_solution_1D():

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

    bc = {
        "left": ["neumann", 0],
        "right": ["neumann", 0]
    }
    bc_params = {
        "left": ["constant", 0],
        "right": ["constant", 0]
    }

    # periodic source term:
    f = lambda x: alpha + beta*np.sin(gamma*x)

    dim = 1

    grid = fem.core.Grid(dim, L, n)

    discretization = fem.core.Discretization()

    source = fem.core.Source(grid, discretization, f)

    diffusion = fem.core.Diffusion(D)

    reaction = fem.core.Reaction(R)

    operators = [diffusion, reaction]
    stiffness = fem.core.StiffnessMatrix(grid, discretization, operators)
    # print(stiffness.s)

    natural_boundary = fem.core.NaturalBoundary(grid, discretization, operators, bc)

    # specify points at which to return function:
    x = np.linspace(0, L, n)

    solution = fem.core.Solution(grid, discretization, bc, stiffness, source, natural_boundary, x)

    assert np.square(solution.u-reference_solution).max() < 10**-8


# if __name__ == '__main__':
#     test_solution_1D()
