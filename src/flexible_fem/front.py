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
            u, x = self.steady_diffusion_reaction_1D(bc_types, bc_params, grid_params,
                                                     core_params, source_params)
        elif pde == 'steady_advection_diffusion_reaction_1D':
            u, x = self.steady_advection_diffusion_reaction_1D(bc_types, bc_params, grid_params,
                                                               core_params, source_params)
        elif pde == 'steady_advection_diffusion_1D':
            u, x = self.steady_advection_diffusion_1D(bc_types, bc_params, grid_params,
                                                      core_params, source_params)
        elif pde == 'laplace_1D':
            u, x = self.laplace_1D(bc_types, bc_params, grid_params, core_params, source_params)
        elif pde == 'laplace_2D':
            u, x = self.laplace_2D(bc_types, bc_params, grid_params, core_params, source_params)
        return u, x

    def steady_diffusion_reaction_1D(self, bc_types, bc_params, grid_params, core_params, source_params):
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

        dim = 1

        grid = core.Grid(dim, L, n)

        discretization = core.Discretization(dim)

        source = core.Source(grid, discretization, f)

        diffusion = core.Diffusion(grid, discretization, bc_types, bc_params, D)

        reaction = core.Reaction(grid, discretization, bc_types, bc_params, R)

        operators = [diffusion, reaction]
        stiffness = core.SolutionOperator(grid, discretization, operators)
        # print(stiffness.s)

        natural_boundary = core.NaturalBoundary(grid, discretization, bc_types, bc_params, operators)

        # specify points at which to return function:
        x_vec = np.linspace(0, L, n)
        xy = [[x] for x in x_vec]

        solution = core.Solution(grid, discretization, bc_types, bc_params, stiffness, source, natural_boundary, xy)
        u = solution.u

        return u, x_vec

    def steady_advection_diffusion_reaction_1D(self, bc_types, bc_params, grid_params,
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

        # weak form:
        # \int_0^L A*(du/dx)*v dx - [D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx
        # + \int_0^L R*u*v dx = \int_0^L f*v dx

        dim = 1

        grid = core.Grid(dim, L, n)

        discretization = core.Discretization(dim)

        source = core.Source(grid, discretization, f)
        # print(source.d)

        advection = core.Advection(grid, discretization, bc_types, bc_params, A)

        diffusion = core.Diffusion(grid, discretization, bc_types, bc_params, D)

        reaction = core.Reaction(grid, discretization, bc_types, bc_params, R)

        operators = [advection, diffusion, reaction]
        stiffness = core.SolutionOperator(grid, discretization, operators)
        # print(stiffness.s)

        natural_boundary = core.NaturalBoundary(grid, discretization, bc_types, bc_params, operators)

        # specify points at which to return function:
        x_vec = np.linspace(0, L, n)
        xy = [[x] for x in x_vec]

        solution = core.Solution(grid, discretization, bc_types, bc_params, stiffness, source, natural_boundary, xy)
        u = solution.u

        return u, x_vec

    def steady_advection_diffusion_1D(self, bc_types, bc_params, grid_params,
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

        # weak form:
        # \int_0^L A*(du/dx)*v dx - [D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx
        # + \int_0^L R*u*v dx = \int_0^L f*v dx

        dim = 1

        grid = core.Grid(dim, L, n)

        discretization = core.Discretization(dim)

        source = core.Source(grid, discretization, f)
        # print(source.d)

        advection = core.Advection(grid, discretization, bc_types, bc_params, A)

        diffusion = core.Diffusion(grid, discretization, bc_types, bc_params, D)

        operators = [advection, diffusion]
        stiffness = core.SolutionOperator(grid, discretization, operators)
        # print(stiffness.s)

        natural_boundary = core.NaturalBoundary(grid, discretization, bc_types, bc_params, operators)

        # specify points at which to return function:
        x_vec = np.linspace(0, L, n)
        xy = [[x] for x in x_vec]

        solution = core.Solution(grid, discretization, bc_types, bc_params, stiffness, source, natural_boundary, xy)
        u = solution.u

        return u, x_vec

    def laplace_1D(self, bc_types, bc_params, grid_params, core_params, source_params):
        # Laplace equation:
        # - D*u_xx = 0
        D = core_params["D"]
        L = grid_params["L"]
        n = grid_params["n"]

        # zero source term:
        f = lambda xy: 0

        # weak form:
        # - [D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx  = 0

        dim = 1

        grid = core.Grid(dim, L, n)

        discretization = core.Discretization(dim)

        source = core.Source(grid, discretization, f)

        diffusion = core.Diffusion(grid, discretization, bc_types, bc_params, D)

        operators = [diffusion]
        stiffness = core.SolutionOperator(grid, discretization, operators)
        # print(stiffness.s)

        natural_boundary = core.NaturalBoundary(grid, discretization, bc_types, bc_params, operators)

        # specify points at which to return function:
        x_vec = np.linspace(0, L, n)
        xy = [[x] for x in x_vec]

        solution = core.Solution(grid, discretization, bc_types, bc_params, stiffness, source, natural_boundary, xy)
        u = solution.u

        return u, x_vec

    def laplace_2D(self, bc_types, bc_params, grid_params, core_params, source_params):
        # Laplace equation:
        # - D*(u_xx + u_yy) = 0
        D = core_params["D"]
        L = grid_params["L"]
        nx = grid_params["nx"]
        H = grid_params["H"]
        ny = grid_params["ny"]

        # zero source term:
        f = lambda xy: 0

        # weak form:
        # \int_S - D*(n_x*u_x + n_y*u_y)*v + \int_V D*(u_x*v_x + u_y*v_y) dxdy  = 0

        dim = 2

        grid = core.Grid(dim, L, nx, H, ny)

        discretization = core.Discretization(dim)

        source = core.Source(grid, discretization, f)

        diffusion = core.Diffusion(grid, discretization, bc_types, bc_params, D)

        operators = [diffusion]
        stiffness = core.SolutionOperator(grid, discretization, operators)
        # print(stiffness.s)

        natural_boundary = core.NaturalBoundary(grid, discretization, bc_types, bc_params, operators)

        # specify points at which to return function:
        x_vec = np.linspace(0, L, nx)
        y_vec = np.linspace(0, H, ny)
        xy = [[x, y] for x, y in zip(x_vec, y_vec)]
        X, Y = np.meshgrid(x_vec, y_vec)

        solution = core.Solution(grid, discretization, bc_types, bc_params, stiffness, source, natural_boundary, xy)
        u = solution.u

        return u, xy
