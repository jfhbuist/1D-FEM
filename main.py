#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:00:57 2022

@author: jurriaan

"""

import numpy as np
# from scipy.integrate import quad
import matplotlib.pyplot as plt
import fem_solver as fem
import sympy as sp

def broadcast(fun):
    # this is needed when using lambdify, 
    # to make a constant function return a vector when the input is a vector
    return lambda *x: np.broadcast_arrays(fun(*x), *x)[0]
    
def plot_dressing_up(n, D, R, alpha, beta, gamma):
    plt.xlim(0,1)
    plt.xlabel('x')
    plt.ylabel('u')
    title = r'n={:d}, D={:.1f}, R={:.1f}, $\alpha$={:.1f}, $\beta$={:.1f}, $\gamma$={:.1f}'.format(n, D, R, alpha, beta, gamma)
    plt.title(title, y=1.05, fontsize = 14)
    plt.legend()
    plt.grid()
    plt.show()
    
class ExactSolution:
    def get_solution(self, pde, bc, n, D, R, alpha, beta, gamma):
        if pde == 'steady_diffusion_reaction_1D':
            u_sym, x_sym = self.steady_diffusion_reaction_1D(bc, D, R, alpha, beta, gamma)
            # convert to a python function:
            u_num = broadcast(sp.lambdify(x_sym, u_sym, "numpy")) 
            # specify points at which to return function:
            x = np.linspace(0,1,n) 
            # evalute function at specified points: 
            u = u_num(x) 
        return u, x
            
    def steady_diffusion_reaction_1D(self, bc, D, R, alpha, beta, gamma):
        # x is our symbolic spatial variable:
        x = sp.symbols("x")
        # periodic source terms:
        f = alpha + beta*sp.sin(gamma*x)
        
        # complementary solution (general solution to the homogeneous equation):
        # uc = c0*exp(mu*x) + c1*exp(-mu*x)  
        mu = np.sqrt(R/D) 
        # particular solution (this may be any solution to the full inhomogeneous equation, and is source dependent):
        up = alpha/R + beta*sp.sin(gamma*x)/(R + D*gamma**2)
        dupdx = sp.diff(up,x) # needed for neumann boundary conditions
        # general solution is the sum of the two:
        # u = uc + up
        # constants are set afterwards through boundary conditions
        
        # we need to solve for c1 and c2 using the boundary conditions
        # we will get an equation of the form A*c = B, with c = [c0; c1].
        A = np.zeros((2,2))
        B = np.zeros(2)
        if bc["left"][0] == "neumann":
            # mu*c0 - mu*c1 + dupdx(0) = bc(left)
            A[0,0] = mu
            A[0,1] = -mu
            B[0] = -dupdx.subs(x,0) + bc["left"][1]
        if bc["right"][0] == "neumann":
            # mu*c0*exp(mu*L) - mu*c1*exp(-mu*L) + dupdx(1) = bc(right)
            L = 1
            A[1,0] = mu*np.exp(mu*L)
            A[1,1] = -mu*np.exp(-mu*L) 
            B[1] = -dupdx.subs(x,L) + bc["right"][1]
        c = np.linalg.solve(A, B)
        c0 = c[0]
        c1 = c[1]
        uc = c0*sp.exp(mu*x) + c1*sp.exp(-mu*x)  
        u = uc + up # this is a symbolic function
        return u, x
    
class NumericalSolution:
    def get_solution(self, pde, bc, n, D, R, alpha, beta, gamma):
        if pde == 'steady_diffusion_reaction_1D':
            u, x = self.steady_diffusion_reaction_1D(bc, n, D, R, alpha, beta, gamma)
        return u, x
    
    def steady_diffusion_reaction_1D(self, bc, n, D, R, alpha, beta, gamma):
        # Diffusion-reaction equation:
        # -D*u_xx + R*u = f
        # with source term:
        # f = alpha + beta*sin(gamma*x)
        # and boundary conditions:
        # -D*u_x(0) = 0, -D*u_x(1) = 0

        # weak form:
        # \int_0^1 D*(du/dx)*(dphi/dx) dx + \int_0^1 R*u*phi dx = \int_0^1 f*phi dx

        # S_{i,j} = \int_0^1 D*(dphi_i/dx)*(dphi_j/dx) dx + \int_0^1 R*phi_i*phi_j dx
        # d_i = \int_0^1 f*phi_i dx        
        
        grid = fem.Grid(n)
        
        discretization = fem.Discretization()
      
        source = fem.Source(grid, discretization, alpha, beta, gamma)
        # print(source.d)
        
        diffusion = fem.Diffusion(grid, discretization, D)
        # print(diffusion.s_D)
      
        reaction = fem.Reaction(grid, discretization, R)
        # print(reaction.s_R)
      
        stiffness = fem.StiffnessMatrix(grid, discretization)
        stiffness.combine_operators([diffusion, reaction])
        # print(stiffness.s)
      
        # solve for solution values at vertices
        # these are actually the coefficients associated with the basis 
        # functions centered at each grid point
        coeffs = np.linalg.solve(stiffness.s,source.d)      
        
        # specify points at which to return function:
        x = np.linspace(0,1,n) 
        u = discretization.construct_solution(coeffs, grid, x)

        return u, x
        
  
def main():

    ## Input
 
    # Reference input:
    # n = 5
    # D = 1
    # R = 0.8
    # alpha = 0.5
    # beta = 2
    # gamma = 30
    
    # set 12
    # n = 100
    # D = 1
    # R = 1
    # alpha = 1
    # beta = 0
    # gamma = 0
    
    # set 13
    # n = 100
    # D = 1
    # R = 1
    # alpha = 0
    # beta = 1
    # gamma = 20
    
    n = 100
    D = 1
    R = 1
    alpha = 0
    beta = 1
    gamma = 10
    
    pde = "steady_diffusion_reaction_1D"
    bc = {
    "left": ["neumann", 0],
    "right": ["neumann", 0]
    }
    u_exact, x_exact = ExactSolution().get_solution(pde, bc, n, D, R, alpha, beta, gamma)
      
    u_fem, x_fem = NumericalSolution().get_solution(pde, bc, n, D, R, alpha, beta, gamma)
    
    plt.plot(x_fem, u_fem, linewidth = 5, label = 'fem')
    plt.plot(x_exact, u_exact, linestyle = ':', linewidth = 5, label = 'exact')
    
    plot_dressing_up(n, D, R, alpha, beta, gamma)
      
if __name__=='__main__':
    main()
  

