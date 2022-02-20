#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:00:57 2022

@author: jurriaan

Diffusion-reaction equation:
-D*u_xx + lambda*u = f
with source term:
f = alpha + a*sin(gamma*x)
and boundary conditions:
-D*u_x(0) = 0, -D*u_x(1) = 0

weak form:
\int_0^1 D*(du/dx)*(dphi/dx) dx + \int_0^1 lambda*u*phi dx = \int_0^1 f*phi dx

S_{i,j} = \int_0^1 D*(dphi_i/dx)*(dphi_j/dx) dx + \int_0^1 lambda*phi_i*phi_j dx
d_i = \int_0^1 f*phi_i dx
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import fem_solver as fem


def plot_solution(x, u, n, D, a, alpha, beta, gamma):
  plt.plot(x,u)
  plt.xlim(0,1)
  plt.xlabel('x')
  plt.ylabel('u')
  title = r'n={:d}, D={:.1f}, a={:.1f}, $\alpha$={:.1f}, $\beta$={:.1f}, $\gamma$={:.1f}'.format(n,D,a, alpha, beta, gamma)
  plt.title(title, y=1.02, fontsize = 16)
  plt.grid()
  plt.show()
  
def main():

  ## Input
  
  # Reference input
  n = 5
  D = 1
  a = 0.8
  alpha = 0.5
  beta = 2
  gamma = 30
  
  # n = 100
  # D = 1
  # a = 1
  # alpha = 0
  # beta = 1
  # gamma = 20
  
  grid = fem.Grid(n)
  
  source = fem.Source(grid, alpha, beta, gamma)
  print(source.d)
    
  diffusion = fem.Diffusion(grid, D)
  print(diffusion.s_D)
  
  advection = fem.Advection(grid, a)
  print(advection.s_A)
  
  stiffness = fem.StiffnessMatrix(grid,[diffusion.s_D,advection.s_A])
  print(stiffness.s)
  
  u = np.linalg.solve(stiffness.s,source.d) # solve for solution values at vertices
  print(u)
  
  plot_solution(grid.x_vert, u, n, D, a, alpha, beta, gamma)
      
if __name__=='__main__':
  main()
  

