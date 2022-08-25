# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:32:48 2022

@author: jurri
"""

import numpy as np
import fem_core as fem
    
class NumericalSolution:
    def get_solution(self, pde, bc, L, n, params):
        if pde == 'steady_diffusion_reaction_1D':
            u, x = self.steady_diffusion_reaction_1D(bc, L, n, params["D"], params["R"], params["alpha"], params["beta"], params["gamma"])
        if pde == 'steady_advection_diffusion_reaction_1D':
            u, x = self.steady_advection_diffusion_reaction_1D(bc, L, n, params["A"], params["D"], params["R"], params["alpha"], params["beta"], params["gamma"])
        if pde == 'steady_advection_diffusion_1D':
            u, x = self.steady_advection_diffusion_1D(bc, L, n, params["A"], params["D"], params["alpha"], params["beta"], params["gamma"])
        return u, x
    
    def steady_diffusion_reaction_1D(self, bc, L, n, D, R, alpha, beta, gamma):
        # Diffusion-reaction equation (aka Helmholtz equation):
        # -D*u_xx + R*u = f
        # with source term:
        # f = alpha + beta*sin(gamma*x)

        # weak form:
        # -[D*(du/dx)*phi]_0^L + \int_0^L D*(du/dx)*(dphi/dx) dx + \int_0^L R*u*phi dx = \int_0^L f*phi dx
        # here phi represents the test function, aka weighting function (equal to basis function in continuous Galerkin)

        # S_{i,j} = \int_0^L D*(dphi_i/dx)*(dphi_j/dx) dx + \int_0^L R*phi_i*phi_j dx
        # d_i = \int_0^L f*phi_i dx   
        
        grid = fem.Grid(L, n)
        
        discretization = fem.Discretization()
      
        source = fem.Source(grid, discretization, alpha, beta, gamma)
        
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
    
    def steady_advection_diffusion_reaction_1D(self, bc, L, n, A, D, R, alpha, beta, gamma):
        # Advection-diffusion-reaction equation
        # A*u_x - D*u_xx + R*u = f
        # with source term:
        # f = alpha + beta*sin(gamma*x)

        # weak form:
        # \int_0^L A*(du/dx)*phi dx - [D*(du/dx)*phi]_0^L + \int_0^L D*(du/dx)*(dphi/dx) dx + \int_0^L R*u*phi dx = \int_0^L f*phi dx
        # here phi represents the test function, aka weighting function (equal to basis function in continuous Galerkin)
        
        grid = fem.Grid(L, n)
        
        discretization = fem.Discretization()
      
        source = fem.Source(grid, discretization, alpha, beta, gamma)
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
    
    def steady_advection_diffusion_1D(self, bc, L, n, A, D, alpha, beta, gamma):
        # Advection-diffusion equation
        # A*u_x - D*u_xx = f
        # with source term:
        # f = alpha + beta*sin(gamma*x)

        # weak form:
        # \int_0^L A*(du/dx)*phi dx - [D*(du/dx)*phi]_0^L + \int_0^L D*(du/dx)*(dphi/dx) dx + \int_0^L R*u*phi dx = \int_0^L f*phi dx
        # here phi represents the test function, aka weighting function (equal to basis function in continuous Galerkin) 
        
        grid = fem.Grid(L, n)
        
        discretization = fem.Discretization()
      
        source = fem.Source(grid, discretization, alpha, beta, gamma)
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