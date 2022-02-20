#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:50:37 2021

@author: jurriaan
"""

"""
FEM solver 1D

Creates stiffness matrix S and source vector d, to set up the problem
  S*u = d

Split domain integral (\int_0^1) into integrals over the elements e_k

Continuous Galerkin: Test functions equal to basis functions
"""

import numpy as np
from scipy.integrate import quad

class Grid:
  def __init__(self,n):
    self.n = n
    self.x_vert, self.x_elem, self.dx_elem = self.generate_mesh()
    self.elmat = self.generate_topology()
  
  def generate_mesh(self):
    n = self.n
    x_vert = np.linspace(0,1,n)
    x_elem = (x_vert[0:n-1]+x_vert[1:n])/2
    #dx = 1/(n-1)
    dx_elem = np.diff(x_vert) # width of each element
    return x_vert, x_elem, dx_elem
    
  def generate_topology(self):
    n = self.n
    # This matrix lists the vertices of each element.
    # ie for element i, evaluate elmat(i), to list the two vertices which
    # border element i
    elmat = np.zeros((n-1,2)).astype(int)
    for idx, element in enumerate(elmat):
      elmat[idx,0] = idx
      elmat[idx,1] = idx + 1
    return elmat
  
class DiscreteOperator:
  def __init__(self, grid):
    self.grid = grid
   
  def define_basis_functions(self, x_i, dx_i):
    # Define basis functions on element i (actually half of the basis function 
    # associated with a vertex).
    # x_i = middle of element.
    x0 = x_i - dx_i/2  # location of left boundary vertex
    x1 = x_i + dx_i/2  # location of right boundary vertex
    # for a given element, phi0 is 1 at the left boundary vertex
    # for a given element, phi1 is 1 at the right boundary vertex
    # define as python function
    phi0 = lambda x : (x1 - x )/dx_i 
    phi1 = lambda x : (x  - x0)/dx_i 
    basis_functions = phi0, phi1
    return basis_functions 
    
class Source(DiscreteOperator):
  def __init__(self, grid, alpha, beta, gamma):
    super().__init__(grid)
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.f = self.generate_periodic_source()
    self.d = self.assemble_source_vector()
    
  def generate_periodic_source(self):
    # define f as a python function
    f = lambda x : self.alpha + self.beta*np.sin(self.gamma*x)
    return f
  
  def generate_integrand(self, test_function_xi, x_i, dx_i):
    f = self.f
    x0 = x_i - dx_i/2
    x1 = x_i + dx_i/2
    f_xi = lambda xi : f(xi*dx_i+x0) # transform f to function of xi
    integrand = lambda xi : f_xi(xi)*test_function_xi(xi)*dx_i
    return integrand
  
  def assemble_source_vector(self):
    # Operates on vertices. For each vertex, some the contributions of all its
    # neighbouring elements.
    grid = self.grid
    d = np.zeros(grid.n)
    # loop over elements
    for i, x_i in enumerate(grid.x_elem): # x_i is center of current element
      dx_i = grid.dx_elem[i] # get width of current element
      d_elem =  self.generate_element_vector(x_i, dx_i)    
      for j in range(grid.elmat.shape[1]):
        # at vertex i we have contributions phi1*f_i + phi0*f_i
        d[grid.elmat[i,j]] += d_elem[j]
    return d
  
  def generate_element_vector(self, x_i, dx_i):
    # Operates on elements.
    x0 = x_i - dx_i/2
    x1 = x_i + dx_i/2
    # get basis functions
    basis_functions = self.define_basis_functions(x_i, dx_i)
    d_elem = np.zeros(len(basis_functions))
    for idx0, test_function in enumerate(basis_functions): # test function
      # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
      test_function_xi = lambda xi : test_function(xi*dx_i+x0) # transform test function to function of xi
      # we integrate over xi, so multiply by dx to get integral over x
      integrand = self.generate_integrand(test_function_xi, x_i, dx_i)
      # integrate over element and put in element vector
      d_elem[idx0] = quad(integrand,0,1)[0] 
    return d_elem

  
class StiffnessMatrix(DiscreteOperator):
  def __init__(self, grid, operators):
    super().__init__(grid)
    self.s = sum(operators)
    
  def assemble_stiffness_matrix(self):
    # Operates on vertices. For each vertex, sum the contributions of all its
    # neighbouring elements. Each vertex has an accompanying linear basis 
    # function, composed of phi1 operating on its left element, and phi0 
    # operating on its right element.
    grid = self.grid
    s = np.zeros((grid.n,grid.n)) # n = number of vertices
    # loop over elements 
    for i, x_i in enumerate(grid.x_elem): # x_i is center of current element
      dx_i = grid.dx_elem[i] # get width of current element
      s_elem = self.generate_element_matrix(x_i, dx_i)
      for j in range(grid.elmat.shape[1]): # loop over test functions
        for k in range(grid.elmat.shape[1]): # loop over solution basis functions
          # at vertex i we have contributions phi1*phi0*u_{i-1} + phi1*phi1*u_i
          # + phi0*phi0*u_i + phi0*phi1*u_{i+1}. 
          s[grid.elmat[i,j],grid.elmat[i,k]] += s_elem[j,k]
    return s

  def generate_element_matrix(self, x_i, dx_i):
    # Element matrix, operates on elements.  
    # Matrix should be symmetric. Numerical calculation.
    # get vertex coordinates
    x0 = x_i - dx_i/2  # location of left boundary vertex
    x1 = x_i + dx_i/2  # location of right boundary vertex
    # get basis functions
    basis_functions = self.define_basis_functions(x_i, dx_i) 
    # calculate s_ij
    s_elem = np.zeros((len(basis_functions),len(basis_functions)))
    for idx0, test_function in enumerate(basis_functions): # test function
      for idx1, basis_function in enumerate(basis_functions): # solution basis function
        # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
        test_function_xi = lambda xi : test_function(xi*dx_i+x0) # transform test function to function of xi
        basis_function_xi = lambda xi : basis_function(xi*dx_i+x0) # transform basis function to function of xi
        integrand = self.generate_integrand(test_function_xi, basis_function_xi, x_i, dx_i)
        # integrate over element and put in element matrix
        s_elem[idx0,idx1] = quad(integrand,0,1)[0] 
    return s_elem
  
class Diffusion(StiffnessMatrix):
  def __init__(self, grid, D):
    self.grid = grid
    self.D = D
    self.s_D = self.assemble_stiffness_matrix()
    
  def generate_integrand(self, test_function_xi, basis_function_xi, x_i, dx_i):
    # generate integrand for diffusion
    # calculate numerical derivative of function using central scheme
    ddx = lambda func, x, dx : (func(x+dx/4)-func(x-dx/4))/(dx/2)
    # multiply ddxi by (1/dx) to get ddx (as function of xi)
    # to take the derivative to xi we need the following:
    dxi = dx_i/dx_i # = 1
    # we integrate over xi, so multiply by dx to get integral over x
    integrand = lambda xi : (self.D*ddx(test_function_xi,xi,dxi)*(1/dx_i)*ddx(basis_function_xi,xi,dxi)*(1/dx_i))*dx_i
    return integrand
    
class Advection(StiffnessMatrix):
  def __init__(self, grid, a):
    self.grid = grid
    self.a = a
    self.s_A = self.assemble_stiffness_matrix()
    
  def generate_integrand(self, test_function_xi, basis_function_xi, x_i, dx_i):
    # generate integrand for advection
    # we integrate over xi, so multiply by dx to get integral over x
    integrand = lambda xi : (self.a*test_function_xi(xi)*basis_function_xi(xi))*dx_i
          # integrate over element and put in element matrix
    return integrand

  