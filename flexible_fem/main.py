#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:00:57 2022

@author: jurriaan

"""

# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

import exact as exact
import fem_front as femf

    
def plot_formatting_1D(ax, pde, bc, grid_params):
    L = grid_params["L"]
    n = grid_params["n"]
    ax.set_xlim(0,L)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    # title = r'n={:d}, A={:.1f}, D={:.1f}, R={:.1f}, $\alpha$={:.1f}, $\beta$={:.1f}, $\gamma$={:.1f}'.format(n, A, D, R, alpha, beta, gamma)
    title = pde
    ax.set_title(title, y=1.05, fontsize = 14)
    ax.legend()
    ax.grid()
    plt.show()
    
def plot_formatting_2D(ax, pde, bc, grid_params):
    L = grid_params["L"]
    H = grid_params["H"]
    nx = grid_params["nx"]
    ny = grid_params["ny"]
    ax.set_xlim(0,L)
    ax.set_ylim(0,H)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    title = pde
    ax.set_title(title, y=1.05, fontsize = 14)
    plt.show()
        
  
def main():

    ### Input
 
    ## Reference input:
    # pde = "steady_diffusion_reaction_1D"
    # bc = {
    # "left": ["neumann", 0],
    # "right": ["neumann", 0]
    # } 
    # bc_params = {
    #     "left": ["constant", 0],
    #     "right": ["constant", 0]
    # }
    # grid_params = {
    #     "L": 1,
    #     "n": 5
    # }
    # core_params = {
    #     "D":        1,
    #     "R":        0.8
    # }
    # source_params = {
    #     "function": "periodic",
    #     "alpha":    0.5,
    #     "beta" :    2,
    #     "gamma":    30
    # }
    
    ## set 12
    # pde = "steady_diffusion_reaction_1D"
    # bc = {
    # "left": ["neumann", 0],
    # "right": ["neumann", 0]
    # } 
    # bc_params = {
    #     "left": ["constant", 0],
    #     "right": ["constant", 0]
    # }
    # grid_params = {
    #     "L": 1,
    #     "n": 100
    # }
    # core_params = {
    #     "D":        1,
    #     "R":        1
    # }
    # source_params = {
    #     "function": "periodic",
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
    # bc_params = {
    #     "left": ["constant", 0],
    #     "right": ["constant", 0]
    # }
    # grid_params = {
    #     "L": 1,
    #     "n": 100
    # }
    # core_params = {
    #     "D":        1,
    #     "R":        1
    # }
    # source_params = {
    #     "function": "periodic",
    #     "alpha":    0,
    #     "beta" :    1,
    #     "gamma":    20
    # }
    
    ## Random
    # pde = "steady_diffusion_reaction_1D"
    # bc = {
    # "left": ["neumann", 0],
    # "right": ["neumann", 0]
    # }
    # bc_params = {
    #     "left": ["constant", 0],
    #     "right": ["constant", 0]
    # }
    # grid_params = {
    #     "L": 1.5,
    #     "n": 100
    # }
    # core_params = {
    #     "D":        2,
    #     "R":        4
    # }
    # source_params = {
    #     "function": "periodic",
    #     "alpha":    0.5,
    #     "beta" :    3,
    #     "gamma":    10
    # }
    
    ## classic advection-diffusion
    # pde = "steady_advection_diffusion_1D"
    # bc = {
    # "left": ["dirichlet", 0],
    # "right": ["dirichlet", 1]
    # }
    # bc_params = {
    #     "left": ["constant", 0],
    #     "right": ["constant", 1]
    # }
    # grid_params = {
    #     "L": 1,
    #     "n": 100
    # }
    # core_params = {
    #     "A":        1,
    #     "D":        0.01,
    #     "R":        0
    # }
    # source_params = {
    #     "function": "periodic",
    #     "alpha":    0,
    #     "beta" :    0,
    #     "gamma":    20 
    # }
    
    ## test all parameters
    # pde = "steady_advection_diffusion_reaction_1D"
    # bc = {
    # "left": ["dirichlet", 1],
    # "right": ["neumann", 0]
    # }
    # bc_params = {
    #     "left": ["constant", 1],
    #     "right": ["constant", 0]
    #     }
    # grid_params = {
    #     "L": 1.7,
    #     "n": 139
    # }
    # core_params = {
    #     "A":        0.5,
    #     "D":        0.01,
    #     "R":        1.3
    # }
    # source_params = {
    #     "function": "periodic",
    #     "alpha":    0.8,
    #     "beta":     3.5,
    #     "gamma":    30        
    #     }
    
    # pde = "laplace_1D"
    # bc = {
    # "left": ["dirichlet", 2],
    # "right": ["neumann", -1]
    # }
    # bc_params = {
    #     "left": ["constant", 1],
    #     "right": ["constant", 0]
    #     }
    # grid_params = {
    #     "L": 1.7,
    #     "n": 139
    # }
    # core_params = {
    #     "D":        1.5,
    # }
    # source_params = {
    #     "function": "zero"      
    #     }
    
    pde = "laplace_2D"
    bc = {
    "left": ["dirichlet"],
    "right": ["dirichlet"],
    "upper": ["dirichlet"],
    "lower": ["dirichlet"]
    }
    grid_params = {
        "L": 1,
        "H": 1,
        "nx": 100,
        "ny": 100
    }
    # bc_params = {
    #     "left": ["quadratic", 1, 0, 0, -(1/(grid_params["H"]/2)**2), grid_params["H"]/2],
    #     "right": ["quadratic", 1, 0, 0, -(1/(grid_params["H"]/2)**2), grid_params["H"]/2],
    #     "upper": ["quadratic", 0, 0, 0, 0, 0],
    #     "lower": ["quadratic", 0, 0, 0, 0, 0]
    #     }
    # bc_params = {
    #     "left": ["quadratic", 1, 0, 0, -(1/(grid_params["H"]/2)**2), grid_params["H"]/2],
    #     "right": ["quadratic", 0, 0, 0, 0, 0],
    #     "upper": ["quadratic", 0, 0, 0, 0, 0],
    #     "lower": ["quadratic", 0, 0, 0, 0, 0]
    #     }
    # bc_params = {
    #     "left": ["quadratic", 0, 0, 0, 0, 0],
    #     "right": ["quadratic", 0, 0, 0, 0, 0],
    #     "upper": ["quadratic", 1, 0, 0, -(1/(grid_params["L"]/2)**2), grid_params["L"]/2],
    #     "lower": ["quadratic", 1, 0, 0, -(1/(grid_params["L"]/2)**2), grid_params["L"]/2]
    #     }
    bc_params = {
        "left": ["sine", 0, 1, np.pi/(grid_params["H"]), 0, 0],
        "right": ["sine", 0, 1, np.pi/(grid_params["H"]), 0, 0],
        "upper": ["sine", 0, 0, 0, 0, 0],
        "lower": ["sine", 0, 0, 0, 0, 0]
    }
    # bc_params = {
    #     "left": ["sine", 0, 0, 0, 0, 0],
    #     "right": ["sine", 0, 0, 0, 0, 0],
    #     "upper": ["sine", 0, 1, np.pi/(grid_params["L"]), 0, 0],
    #     "lower": ["sine", 0, 1, np.pi/(grid_params["L"]), 0, 0],
    #     }
    # bc_params = {
    #     "left": ["cosine", 1, -1, np.pi/(grid_params["H"]/2), 0, 0], 
    #     "right": ["cosine", 1, -1, np.pi/(grid_params["H"]/2), 0, 0],
    #     "upper": ["cosine", 0, 0, 0, 0, 0],
    #     "lower": ["cosine", 0, 0, 0, 0, 0]
    #     }
    # bc_params = {
    #     "left": ["cosine", 0, 0, 0, 0, 0],
    #     "right": ["cosine", 0, 0, 0, 0, 0],
    #     "upper": ["cosine", 1, -1, np.pi/(grid_params["L"]/2), 0, 0],
    #     "lower": ["cosine", 1, -1, np.pi/(grid_params["L"]/2), 0, 0],
    #     }
    core_params = {
        "D":        1
    }
    source_params = {
        "function": "zero",
    }

    # u_exact, x_exact = exact.ExactSolution().get_solution(pde, bc, bc_params, grid_params, core_params, source_params)
    # u_fem, x_fem = femf.NumericalSolution().get_solution(pde, bc, bc_params, grid_params, core_params, source_params)
    
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.plot(x_fem, u_fem, linewidth = 5, label = 'fem')
    # ax.plot(x_exact, u_exact, linestyle = ':', linewidth = 5, label = 'exact')
    # plot_formatting_1D(ax, pde, bc, grid_params)
    
    u_exact, x_exact, y_exact = exact.ExactSolution().get_solution(pde, bc, bc_params, grid_params, core_params, source_params)
    fig = plt.figure()   
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_exact, y_exact, u_exact) #, label = 'exact')       
    plot_formatting_2D(ax, pde, bc, grid_params)
    
      
if __name__=='__main__':
    main()
  

