#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:00:57 2022

@author: jurriaan

"""

import matplotlib.pyplot as plt
import exact as exact
import fem_front as femf

    
def plot_dressing_up(pde, bc, L, n, params):
    plt.xlim(0,L)
    plt.xlabel('x')
    plt.ylabel('u')
    # title = r'n={:d}, A={:.1f}, D={:.1f}, R={:.1f}, $\alpha$={:.1f}, $\beta$={:.1f}, $\gamma$={:.1f}'.format(n, A, D, R, alpha, beta, gamma)
    title = pde
    plt.title(title, y=1.05, fontsize = 14)
    plt.legend()
    plt.grid()
    plt.show()
        
  
def main():

    ### Input
 
    ## Reference input:
    # pde = "steady_diffusion_reaction_1D"
    # bc = {
    # "left": ["neumann", 0],
    # "right": ["neumann", 0]
    # } 
    # L = 1
    # n = 5
    # params = {
    #     "D":        1,
    #     "R":        0.8,
    #     "alpha":    0.5,
    #     "beta" :    2,
    #     "gamma":    30,
    # }
    
    ## set 12
    # pde = "steady_diffusion_reaction_1D"
    # bc = {
    # "left": ["neumann", 0],
    # "right": ["neumann", 0]
    # } 
    # L = 1
    # n = 100
    # params = {
    #     "D":        1,
    #     "R":        1,
    #     "alpha":    1,
    #     "beta" :    0,
    #     "gamma":    0,
    # }
    
    ## set 13
    # pde = "steady_diffusion_reaction_1D"
    # bc = {
    # "left": ["neumann", 0],
    # "right": ["neumann", 0]
    # } 
    # L = 1
    # n = 100
    # params = {
    #     "D":        1,
    #     "R":        1,
    #     "alpha":    0,
    #     "beta" :    1,
    #     "gamma":    20,
    # }
    
    ## Random
    # pde = "steady_diffusion_reaction_1D"
    # bc = {
    # "left": ["neumann", 0],
    # "right": ["neumann", 0]
    # }
    # L = 1.5
    # n = 100
    # params = {
    #     "D":        2,
    #     "R":        4,
    #     "alpha":    0.5,
    #     "beta" :    3,
    #     "gamma":    10,
    # }
    
    ## classic advection-diffusion
    # pde = "steady_advection_diffusion_1D"
    # bc = {
    # "left": ["dirichlet", 0],
    # "right": ["dirichlet", 1]
    # }
    # L = 1
    # n = 100
    # params = {
    #     "A":        1,
    #     "D":        0.01,
    #     "R":        0,
    #     "alpha":    0,
    #     "beta" :    0,
    #     "gamma":    20,
    # }
    
    ## test all parameters
    pde = "steady_advection_diffusion_reaction_1D"
    bc = {
    "left": ["dirichlet", 1],
    "right": ["neumann", 0]
    }
    L = 1.7
    n = 139
    params = {
        "A":        0.5,
        "D":        0.01,
        "R":        1.3,
        "alpha":    0.8,
        "beta":     3.5,
        "gamma":    30,
    }
    
    u_exact, x_exact = exact.ExactSolution().get_solution(pde, bc, L, n, params)
    u_fem, x_fem = femf.NumericalSolution().get_solution(pde, bc, L, n, params)
    
    plt.plot(x_fem, u_fem, linewidth = 5, label = 'fem')
    plt.plot(x_exact, u_exact, linestyle = ':', linewidth = 5, label = 'exact')
    
    plot_dressing_up(pde, bc, L, n, params)
    
      
if __name__=='__main__':
    main()
  

