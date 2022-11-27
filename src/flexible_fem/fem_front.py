# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:32:48 2022

@author: jurri
"""

import numpy as np

from . import fem_core as fem
    
class NumericalSolution:
    def get_solution(self, pde, bc, bc_params, grid_params, core_params, source_params):
        if pde == 'steady_diffusion_reaction_1D':
            u, x = self.steady_diffusion_reaction_1D(bc, bc_params, grid_params, core_params, source_params)
        elif pde == 'steady_advection_diffusion_reaction_1D':
            u, x = self.steady_advection_diffusion_reaction_1D(bc, bc_params, grid_params, core_params, source_params)
        elif pde == 'steady_advection_diffusion_1D':
            u, x = self.steady_advection_diffusion_1D(bc, bc_params, grid_params, core_params, source_params)
        elif pde == 'laplace_1D':
            u, x = self.laplace_1D(bc, bc_params, grid_params, core_params, source_params)
        return u, x
    
    def steady_diffusion_reaction_1D(self, bc, bc_params, grid_params, core_params, source_params):
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
            f = lambda x : alpha + beta*np.sin(gamma*x)

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
        # Each vertex has multiple associated test functions which reach a value of 1 at that vertex.
        
        # We use standard (continuous) Galerkin, in which the test functions are equal to the basis functions:
        # v_i = phi_i
        
        dim = 1
        
        grid = fem.Grid(L, n)
        
        discretization = fem.Discretization()
      
        source = fem.Source(grid, discretization, f)
        
        diffusion = fem.Diffusion(D)
      
        reaction = fem.Reaction(R)
      
        operators = [diffusion, reaction]
        stiffness = fem.StiffnessMatrix(grid, discretization, operators)
        # print(stiffness.s)
        
        natural_boundary = fem.NaturalBoundary(grid, discretization, operators, bc)
        
        # specify points at which to return function:
        x = np.linspace(0,L,n) 
        
        solution = fem.Solution(grid, discretization, bc, stiffness, source, natural_boundary, x)
        u = solution.u

        return u, x
    
    def steady_advection_diffusion_reaction_1D(self, bc, bc_params, grid_params, core_params, source_params):
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
            f = lambda x : alpha + beta*np.sin(gamma*x)

        # weak form:
        # \int_0^L A*(du/dx)*v dx - [D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx + \int_0^L R*u*v dx = \int_0^L f*v dx
        
        dim = 1
        
        grid = fem.Grid(L, n)
        
        discretization = fem.Discretization()
      
        source = fem.Source(grid, discretization, f)
        # print(source.d)
        
        advection = fem.Advection(A)
        
        diffusion = fem.Diffusion(D)
      
        reaction = fem.Reaction(R)
      
        operators = [advection, diffusion, reaction]
        stiffness = fem.StiffnessMatrix(grid, discretization, operators)
        # print(stiffness.s)
        
        natural_boundary = fem.NaturalBoundary(grid, discretization, operators, bc)
        
        # specify points at which to return function:
        x = np.linspace(0,L,n) 
        
        solution = fem.Solution(grid, discretization, bc, stiffness, source, natural_boundary, x)
        u = solution.u

        return u, x
    
    def steady_advection_diffusion_1D(self, bc, bc_params, grid_params, core_params, source_params):
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
            f = lambda x : alpha + beta*np.sin(gamma*x)

        # weak form:
        # \int_0^L A*(du/dx)*v dx - [D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx + \int_0^L R*u*v dx = \int_0^L f*v dx
        
        dim = 1
        
        grid = fem.Grid(L, n)
        
        discretization = fem.Discretization()
      
        source = fem.Source(grid, discretization, f)
        # print(source.d)
        
        advection = fem.Advection(A)
        
        diffusion = fem.Diffusion(D)
      
        operators = [advection, diffusion]
        stiffness = fem.StiffnessMatrix(grid, discretization, operators)
        # print(stiffness.s)
        
        natural_boundary = fem.NaturalBoundary(grid, discretization, operators, bc)
        
        # specify points at which to return function:
        x = np.linspace(0,L,n) 
        
        solution = fem.Solution(grid, discretization, bc, stiffness, source, natural_boundary, x)
        u = solution.u

        return u, x
    
    def laplace_1D(self, bc, bc_params, grid_params, core_params, source_params):
        # Laplace equation:
        # - D*u_xx = 0
        D = core_params["D"]
        L = grid_params["L"]
        n = grid_params["n"]
        
        # zero source term:
        f = lambda x : 0

        # weak form:
        # \int_0^L - [D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx  = 0
        
        dim = 1
        
        grid = fem.Grid(L, n)
        
        discretization = fem.Discretization()
      
        source = fem.Source(grid, discretization, f)
        
        diffusion = fem.Diffusion(D)
      
        operators = [diffusion]
        stiffness = fem.StiffnessMatrix(grid, discretization, operators)
        # print(stiffness.s)
        
        natural_boundary = fem.NaturalBoundary(grid, discretization, operators, bc)
        
        # specify points at which to return function:
        x = np.linspace(0,L,n) 
        
        solution = fem.Solution(grid, discretization, bc, stiffness, source, natural_boundary, x)
        u = solution.u

        return u, x