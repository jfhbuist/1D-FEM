# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:32:48 2022

@author: jurri
"""

import numpy as np

from . import core


class NumericalSolution:
    def get_solution(self, pde, bc_types, bc_params, grid_params, core_params, source_params):
        if pde == 'steady_diffusion_reaction_1D':
            dim = 1
            u, x = self.steady_diffusion_reaction_1D(dim, bc_types, bc_params, grid_params,
                                                     core_params, source_params)
        elif pde == 'steady_advection_diffusion_reaction_1D':
            dim = 1
            u, x = self.steady_advection_diffusion_reaction_1D(dim, bc_types, bc_params, grid_params,
                                                               core_params, source_params)
        elif pde == 'steady_advection_diffusion_1D':
            dim = 1
            u, x = self.steady_advection_diffusion_1D(dim, bc_types, bc_params, grid_params,
                                                      core_params, source_params)
        elif pde == 'laplace_1D':
            dim = 1
            u, x = self.laplace_1D(dim, bc_types, bc_params, grid_params, core_params, source_params)
        elif pde == 'laplace_2D':
            dim = 2
            U, X, Y = self.laplace_2D(dim, bc_types, bc_params, grid_params, core_params, source_params)
        if dim == 1:
            return u, x
        elif dim == 2:
            return U, X, Y

    def steady_diffusion_reaction_1D(self, dim, bc_types, bc_params, grid_params, core_params, source_params):
        # Diffusion-reaction equation (aka Helmholtz equation):
        # -D*u_xx + R*u = f
        D = core_params["D"]
        R = core_params["R"]
        L = grid_params["L"]
        n = grid_params["n"]
        source_function = source_params["function"]
        alpha = source_params["alpha"]
        beta = source_params["beta"]
        gamma = source_params["gamma"]

        if source_function == "periodic":
            # periodic source term:
            # xy is a list, with xy[0] = x
            # this is done for generality, so that the code works for 1D and 2D
            f = lambda xy: alpha + beta*np.sin(gamma*xy[0])
            
        bc_functions = {}
        for lb in bc_params:
            if bc_params[lb][0] == "constant":
                bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]

        # weak form:
        # -[D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx + \int_0^L R*u*v dx = \int_0^L f*v dx
        # here v represents the test function, aka weighting function

        # Now let
        # u = \sum_i c_j*phi_j
        # v = \sum v_i

        # The problem to solve is
        # S*c = d
        # S_{i,j} = \int_0^L D*(v_i/dx)*(dphi_j/dx) dx + \int_0^L R*v_i*phi_j dx
        # c_i = c_j
        # d_i = \int_0^L f*v_i dx
        # Each equation in the linear system is associated with one test function v_i.
        # Each vertex has multiple associated test functions which reach a value of 1 at that
        # vertex.

        # We use standard (continuous) Galerkin, in which the test functions are equal to the
        # basis functions:
        # v_i = phi_i = N_i

        grid = core.Grid(dim, L, n)

        discretization = core.Discretization(dim)

        source = core.Source(grid, discretization, f)

        diffusion = core.Diffusion(grid, discretization, bc_types, bc_functions, D)

        reaction = core.Reaction(grid, discretization, bc_types, bc_functions, R)

        operators = [diffusion, reaction]
        stiffness = core.SolutionOperator(grid, discretization, operators)
        # print(stiffness.s)

        natural_boundary = core.NaturalBoundary(grid, discretization, bc_types, bc_functions, operators)

        # specify points at which to return function:
        x_vec = np.linspace(0, L, n)
        xy = [[x] for x in x_vec]

        solution = core.Solution(grid, discretization, bc_types, bc_functions, stiffness, source, natural_boundary, xy)
        u = solution.u

        return u, x_vec

    def steady_advection_diffusion_reaction_1D(self, dim, bc_types, bc_params, grid_params,
                                               core_params, source_params):
        # Advection-diffusion-reaction equation
        # A*u_x - D*u_xx + R*u = f
        A = core_params["A"]
        D = core_params["D"]
        R = core_params["R"]
        L = grid_params["L"]
        n = grid_params["n"]
        source_function = source_params["function"]
        alpha = source_params["alpha"]
        beta = source_params["beta"]
        gamma = source_params["gamma"]

        if source_function == "periodic":
            # periodic source term:
            f = lambda xy: alpha + beta*np.sin(gamma*xy[0])
            
        bc_functions = {}
        for lb in bc_params:
            if bc_params[lb][0] == "constant":
                bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]

        # weak form:
        # \int_0^L A*(du/dx)*v dx - [D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx
        # + \int_0^L R*u*v dx = \int_0^L f*v dx

        grid = core.Grid(dim, L, n)

        discretization = core.Discretization(dim)

        source = core.Source(grid, discretization, f)
        # print(source.d)

        advection = core.Advection(grid, discretization, bc_types, bc_functions, A)

        diffusion = core.Diffusion(grid, discretization, bc_types, bc_functions, D)

        reaction = core.Reaction(grid, discretization, bc_types, bc_functions, R)

        operators = [advection, diffusion, reaction]
        stiffness = core.SolutionOperator(grid, discretization, operators)
        # print(stiffness.s)

        natural_boundary = core.NaturalBoundary(grid, discretization, bc_types, bc_functions, operators)

        # specify points at which to return function:
        x_vec = np.linspace(0, L, n)
        xy = [[x] for x in x_vec]

        solution = core.Solution(grid, discretization, bc_types, bc_functions, stiffness, source, natural_boundary, xy)
        u = solution.u

        return u, x_vec

    def steady_advection_diffusion_1D(self, dim, bc_types, bc_params, grid_params,
                                      core_params, source_params):
        # Advection-diffusion equation
        # A*u_x - D*u_xx = f
        A = core_params["A"]
        D = core_params["D"]
        L = grid_params["L"]
        n = grid_params["n"]
        source_function = source_params["function"]
        alpha = source_params["alpha"]
        beta = source_params["beta"]
        gamma = source_params["gamma"]

        if source_function == "periodic":
            # periodic source term:
            f = lambda xy: alpha + beta*np.sin(gamma*xy[0])
            
        bc_functions = {}
        for lb in bc_params:
            if bc_params[lb][0] == "constant":
                bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]

        # weak form:
        # \int_0^L A*(du/dx)*v dx - [D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx
        # + \int_0^L R*u*v dx = \int_0^L f*v dx

        grid = core.Grid(dim, L, n)

        discretization = core.Discretization(dim)

        source = core.Source(grid, discretization, f)
        # print(source.d)

        advection = core.Advection(grid, discretization, bc_types, bc_functions, A)

        diffusion = core.Diffusion(grid, discretization, bc_types, bc_functions, D)

        operators = [advection, diffusion]
        stiffness = core.SolutionOperator(grid, discretization, operators)
        # print(stiffness.s)

        natural_boundary = core.NaturalBoundary(grid, discretization, bc_types, bc_functions, operators)

        # specify points at which to return function:
        x_vec = np.linspace(0, L, n)
        xy = [[x] for x in x_vec]

        solution = core.Solution(grid, discretization, bc_types, bc_functions, stiffness, source, natural_boundary, xy)
        u = solution.u

        return u, x_vec

    def laplace_1D(self, dim, bc_types, bc_params, grid_params, core_params, source_params):
        # Laplace equation:
        # - D*u_xx = 0
        D = core_params["D"]
        L = grid_params["L"]
        n = grid_params["n"]

        # zero source term:
        f = lambda xy: 0
        
        bc_functions = {}
        for lb in bc_params:
            if bc_params[lb][0] == "constant":
                bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]

        # weak form:
        # - [D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx  = 0

        grid = core.Grid(dim, L, n)

        discretization = core.Discretization(dim)

        source = core.Source(grid, discretization, f)

        diffusion = core.Diffusion(grid, discretization, bc_types, bc_functions, D)

        operators = [diffusion]
        stiffness = core.SolutionOperator(grid, discretization, operators)
        # print(stiffness.s)

        natural_boundary = core.NaturalBoundary(grid, discretization, bc_types, bc_functions, operators)

        # specify points at which to return function:
        x_vec = np.linspace(0, L, n)
        xy = [[x] for x in x_vec]

        solution = core.Solution(grid, discretization, bc_types, bc_functions, stiffness, source, natural_boundary, xy)
        u = solution.u

        return u, x_vec

    def laplace_2D(self, dim, bc_types, bc_params, grid_params, core_params, source_params):
        # Laplace equation:
        # - D*(u_xx + u_yy) = 0
        D = core_params["D"]
        L = grid_params["L"]
        nx = grid_params["nx"]
        H = grid_params["H"]
        ny = grid_params["ny"]

        # zero source term:
        f = lambda xy: 0

        bc_functions = {}
        for lb in bc_params:
            if bc_params[lb][0] == "constant":
                bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1]
            elif bc_params[lb][0] == "sine":
                if lb == "left" or lb == "right":
                    bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1] + bc_params[lb][2]*np.sin(bc_params[lb][3]*xy[1])
                elif lb == "bottom" or lb == "top":
                    bc_functions[lb] = lambda xy, lb=lb: bc_params[lb][1] + bc_params[lb][2]*np.sin(bc_params[lb][3]*xy[0])

        grid = core.Grid(dim, L, nx, H, ny)

        discretization = core.Discretization(dim)

        source = core.Source(grid, discretization, f)

        diffusion = core.Diffusion(grid, discretization, bc_types, bc_functions, D)

        operators = [diffusion]
        stiffness = core.SolutionOperator(grid, discretization, operators)
        # print(stiffness.s)

        natural_boundary = core.NaturalBoundary(grid, discretization, bc_types, bc_functions, operators)

        # specify points at which to return function:
        x_vec = np.linspace(0, L, nx)
        y_vec = np.linspace(0, H, ny)
        xy = np.array([[x, y] for x in x_vec for y in y_vec])
        # X, Y = np.meshgrid(x_vec, y_vec)

        solution = core.Solution(grid, discretization, bc_types, bc_functions, stiffness, source, natural_boundary, xy)
        u = solution.u

        # Create meshgrid and get solution on meshgrid:
        X = np.array([coord[0] for coord in xy]).reshape(x_vec.shape[0], y_vec.shape[0]).transpose()
        Y = np.array([coord[1] for coord in xy]).reshape(x_vec.shape[0], y_vec.shape[0]).transpose()
        U = np.array([sol for sol in u]).reshape(x_vec.shape[0], y_vec.shape[0]).transpose()

        return U, X, Y
