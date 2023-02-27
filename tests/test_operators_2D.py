import numpy as np

import flexible_fem as fem


def test_diffusion_2D():
    # Test the construction of the diffusion operator

    # Load test data
    reference_stiffness = np.loadtxt('tests/data_test_diffusion_2D.txt')
    # To save test data, use
    # "np.savetxt('tests/data_test_diffusion_2D.txt', stiffness.s, fmt='%1.4e')"

    dim = 2
    D = 1.3
    L = 1.2
    nx = 4
    H = 1.5
    ny = 6

    bc_types = {
        "left": "dirichlet",
        "right": "dirichlet",
        "bottom": "dirichlet",
        "top": "dirichlet",
    }
    bc_params = {
        "left": ["sine", 0, 1, np.pi/H, 0, 0],  # g(y) = a + b*sin(c*y)
        "right": ["sine", 0, 1, np.pi/H, 0, 0],  # g(y) = a + b*sin(c*y)
        "bottom": ["sine", 0, 0, 0, 0, 0],  # g(x) = a + b*sin(c*x)
        "top": ["sine", 0, 0, 0, 0, 0],  # g(x) = a + b*sin(c*x)
    }

    bc_functions = {}
    for lb in bc_params:
        if bc_params[lb][0] == "constant":
            bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]
        elif bc_params[lb][0] == "quadratic":
            if lb == "left" or lb == "right":
                bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1] + bc_params[lb][2]*(xy[1]-bc_params[lb][3]) + bc_params[lb][4]*(xy[1]-bc_params[lb][5])**2
            elif lb == "bottom" or lb == "top":
                bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1] + bc_params[lb][2]*(xy[0]-bc_params[lb][3]) + bc_params[lb][4]*(xy[0]-bc_params[lb][5])**2
        elif bc_params[lb][0] == "sine":
            if lb == "left" or lb == "right":
                bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1] + bc_params[lb][2]*np.sin(bc_params[lb][3]*xy[1])
            elif lb == "bottom" or lb == "top":
                bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1] + bc_params[lb][2]*np.sin(bc_params[lb][3]*xy[0])

    grid = fem.core.Grid(dim, L, nx, H, ny)

    discretization = fem.core.Discretization(dim)

    diffusion = fem.core.Diffusion(grid, discretization, bc_types, bc_functions, D)

    operators = [diffusion]
    stiffness = fem.core.SolutionOperator(grid, discretization, operators)
    # print(stiffness.s)

    assert np.square(stiffness.s-reference_stiffness).max() < 10**-8


# For debugging purposes
if __name__ == '__main__':
    test_diffusion_2D()
