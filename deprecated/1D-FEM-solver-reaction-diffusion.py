#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:50:37 2021

@author: jurriaan
"""

"""
Theory

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

S*u = d

Split \int_0^1 into integrals over the elements e_k

Continuous Galerkin: Test functions equal to basis functions
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sympy as sp

def define_source_function(alpha, beta, gamma, mode):
  if mode == 0:
    # define f by parameterization
    f = alpha, beta, gamma
  elif mode == 1:
    # define f as a python function
    f = lambda x : alpha + beta*np.sin(gamma*x)
  elif mode == 2:
    # define f as a symbolic expression
    x = sp.symbols("x")
    f = alpha + beta*sp.sin(gamma*x)
  return f

def generate_mesh(n):
  x_vert = np.linspace(0,1,n)
  x_elem = (x_vert[0:n-1]+x_vert[1:n])/2
  #dx = 1/(n-1)
  dx_elem = np.diff(x_vert) # width of each element
  return x_vert, x_elem, dx_elem

def generate_topology(n):
  # This matrix lists the vertices of each element.
  # ie for element i, evaluate elmat(i), to list the two vertices which
  # border element i
  elmat = np.zeros((n-1,2)).astype(int)
  for idx, element in enumerate(elmat):
    elmat[idx,0] = idx
    elmat[idx,1] = idx + 1
  return elmat
  
def define_basis_functions(x_i, dx_i, mode):
  # Define basis functions on element i (actually half of the basis function 
  # associated with a vertex).
  # x_i = middle of element.
  x0 = x_i - dx_i/2  # location of left boundary vertex
  x1 = x_i + dx_i/2  # location of right boundary vertex
  # for a given element, phi0 is 1 at the left boundary vertex
  # for a given element, phi1 is 1 at the right boundary vertex
  if mode == 0:
    # define by parameterization
    phi0 = x1/dx_i, -1/dx_i
    phi1 = -x0/dx_i, 1/dx_i
  elif mode == 1:
    # define as python function
    phi0 = lambda x : (x1 - x )/dx_i 
    phi1 = lambda x : (x  - x0)/dx_i 
  elif mode == 2:
    # define as symbolic polynomial
    x = sp.symbols("x")
    phi0 = (x1 - x )/dx_i   # 1 - xi
    phi1 = (x  - x0)/dx_i   # xi
  basis_functions = phi0, phi1
  return basis_functions

def generate_element_matrix_analytical(x_i, dx_i, D, a):
  # Element matrix, operates on elements. # Assume linear 1D basis functions. 
  # Matrix should be symmetric. Analytical calculation.
  # diffusion: D*int_0^1 ((d/dxi)test_function_i(xi)*(dxi/dx) * (d/dxi)basis_function_j(xi)*(dxi/dx))*dx_i dxi
  # reaction:  a*int_0^1 test_function_i(xi)*basis_function_j(xi)*dx_i dxi
  s_elem = np.array([[D/dx_i + a*dx_i/3, -D/dx_i + a*dx_i/6], [-D/dx_i + a*dx_i/6, D/dx_i + a*dx_i/3]]) 
  return s_elem
  
def generate_element_matrix_numerical(x_i, dx_i, D, a):
  # get vertex coordinates
  x0 = x_i - dx_i/2  # location of left boundary vertex
  x1 = x_i + dx_i/2  # location of right boundary vertex
  # get basis functions
  basis_functions = define_basis_functions(x_i, dx_i, 1) # mode set to numerical
  # calculate s_ij
  s_elem = np.zeros((len(basis_functions),len(basis_functions)))
  # calculate numerical derivative of function using central scheme
  ddx = lambda func, x, dx : (func(x+dx/4)-func(x-dx/4))/(dx/2)
  for idx0, basis_function_0 in enumerate(basis_functions): # basis_function_0 corresponds to test function
    for idx1, basis_function_1 in enumerate(basis_functions): # basis_function_1 corresponds to solution basis function
      # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
      basis_function_0_xi = lambda xi : basis_function_0(xi*dx_i+x0) # transform basis function to function of xi
      basis_function_1_xi = lambda xi : basis_function_1(xi*dx_i+x0) # transform basis function to function of xi
      # multiply ddxi by (1/dx) to get ddx (as function of xi)
      # to take the derivative to xi we need the following:
      dxi = dx_i/dx_i # = 1
      # we integrate over xi, so multiply by dx to get integral over x
      integrand = lambda xi : (D*ddx(basis_function_0_xi,xi,dxi)*(1/dx_i)*ddx(basis_function_1_xi,xi,dxi)*(1/dx_i) + a*basis_function_0_xi(xi)*basis_function_1_xi(xi))*dx_i
      # integrate over element and put in element matrix
      s_elem[idx0,idx1] = quad(integrand,0,1)[0] 
  return s_elem

def generate_element_matrix_symbolic(x_i, dx_i, D, a):
  # get vertex coordinates
  x0 = x_i - dx_i/2  # location of left boundary vertex
  x1 = x_i + dx_i/2  # location of right boundary vertex
  x = sp.symbols("x") # x is the original symbolic variable of basis functions
  xi = sp.symbols("xi") # we use xi as the new symbolic variable
  # get basis functions
  basis_functions = define_basis_functions(x_i, dx_i, 2) # mode set to symbolic
  # calculate s_ij
  s_elem = np.zeros((len(basis_functions),len(basis_functions)))
  for idx0, basis_function_0 in enumerate(basis_functions): # basis_function_0 corresponds to test function
    for idx1, basis_function_1 in enumerate(basis_functions): # basis_function_1 corresponds to solution basis function
      # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
      basis_function_0_xi = basis_function_0.subs(x,xi*dx_i+x0) # transform basis function to function of xi
      basis_function_1_xi = basis_function_1.subs(x,xi*dx_i+x0) # transform basis function to function of xi
      # multiply ddxi by (1/dx) to get ddx (as function of xi)
      # we integrate over xi, so multiply by dx to get integral over x
      integrand = (D*sp.diff(basis_function_0_xi,xi)*(1/dx_i)*sp.diff(basis_function_1_xi,xi)*(1/dx_i) + a*basis_function_0_xi*basis_function_1_xi)*dx_i
      # integrate over xi to get indefinite integral
      integral = sp.integrate(integrand,xi)
      # compute definite integral over element and put in element matrix
      s_elem[idx0,idx1] = integral.subs(xi,1)-integral.subs(xi,0) 
  return s_elem

def assemble_stiffness_matrix(n, elmat, x_elem, dx_elem, D, a, mode):
  # Operates on vertices. For each vertex, sum the contributions of all its
  # neighbouring elements. Each vertex has an accompanying linear basis 
  # function, composed of phi1 operating on its left element, and phi0 
  # operating on its right element.
  s = np.zeros((n,n)) # n = number of vertices
  # loop over elements 
  for i, x_i in enumerate(x_elem): # x_i is center of current element
    dx_i = dx_elem[i] # get width of current element
    if mode == 0:
      if i ==0: 
        print('Assembling stiffness matrix analytically...')
      s_elem = generate_element_matrix_analytical(x_i, dx_i, D, a)
    elif mode == 1:
      if i ==0: 
        print('Assembling stiffness matrix numerically...')
      s_elem = generate_element_matrix_numerical(x_i, dx_i, D, a)
    elif mode == 2:
      if i ==0: 
        print('Assembling stiffness matrix symbolically...')
      s_elem = generate_element_matrix_symbolic(x_i, dx_i, D, a)
    for j in range(elmat.shape[1]): # loop over test functions
      for k in range(elmat.shape[1]): # loop over solution basis functions
        # at vertex i we have contributions phi1*phi0*u_{i-1} + phi1*phi1*u_i
        # + phi0*phi0*u_i + phi0*phi1*u_{i+1}. 
        s[elmat[i,j],elmat[i,k]] += s_elem[j,k]
  return s
        
def generate_element_vector_analytical(x_i,dx_i,f):
  # Operates on elements.
  x0 = x_i - dx_i/2
  x1 = x_i + dx_i/2
  alpha, beta, gamma = f
  d_elem = np.array([-(beta/gamma)*(np.cos(gamma*x1)-np.cos(gamma*x0))*(1+x0/dx_i) - (beta/dx_i)*(-x1*np.cos(gamma*x1)/gamma + np.sin(gamma*x1)/gamma**2 + x0*np.cos(gamma*x0)/gamma - np.sin(gamma*x0)/gamma**2) + alpha*dx_i/2, -(beta/gamma)*(np.cos(gamma*x1)-np.cos(gamma*x0))*(-x0/dx_i) + (beta/dx_i)*(-x1*np.cos(gamma*x1)/gamma + np.sin(gamma*x1)/gamma**2 + x0*np.cos(gamma*x0)/gamma - np.sin(gamma*x0)/gamma**2) + alpha*dx_i/2]).T # correct  
  return d_elem  

def generate_element_vector_numerical(x_i,dx_i,f):
  # Operates on elements.
  x0 = x_i - dx_i/2
  x1 = x_i + dx_i/2
  f_xi = lambda xi : f(xi*dx_i+x0) # transform f to function of xi
  # get basis functions
  basis_functions = define_basis_functions(x_i, dx_i, 1)
  d_elem = np.zeros(len(basis_functions))
  for idx0, basis_function_0 in enumerate(basis_functions): # basis_function_0 corresponds to test function
    # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
    basis_function_0_xi = lambda xi : basis_function_0(xi*dx_i+x0) # transform basis function to function of xi
    # we integrate over xi, so multiply by dx to get integral over x
    integrand = lambda xi : f_xi(xi)*basis_function_0_xi(xi)*dx_i
    # integrate over element and put in element vector
    d_elem[idx0] = quad(integrand,0,1)[0] 
  return d_elem

def generate_element_vector_symbolic(x_i,dx_i,f):
  # Operates on elements.
  x0 = x_i - dx_i/2
  x1 = x_i + dx_i/2
  x = sp.symbols("x") # x is the original symbolic variable of f
  xi = sp.symbols("xi") # we use xi as the new symbolic variable
  f_xi = f.subs(x,xi*dx_i+x0) # transform f to function of xi
  # get basis functions
  basis_functions = define_basis_functions(x_i, dx_i, 2)
  d_elem = np.zeros(len(basis_functions))
  for idx0, basis_function_0 in enumerate(basis_functions): # basis_function_0 corresponds to test function
    # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
    basis_function_0_xi = basis_function_0.subs(x,xi*dx_i+x0) # transform basis function to function of xi
    # we integrate over xi, so multiply by dx to get integral over x
    integrand = f_xi*basis_function_0_xi*dx_i
    # integrate over xi to get indefinite integral
    integral = sp.integrate(integrand,xi)
    # integrate over element and put in element vector
    d_elem[idx0] = integral.subs(xi,1)-integral.subs(xi,0) 
  return d_elem

def assemble_source_vector(n,elmat,x_elem,dx_elem,f,mode):
  # Operates on vertices. For each vertex, some the contributions of all its
  # neighbouring elements.
  d = np.zeros(n)
  # loop over elements
  for i, x_i in enumerate(x_elem): # x_i is center of current element
    dx_i = dx_elem[i] # get width of current element
    if mode == 0:
      if i ==0: 
        print('Assembling source vector analytically...')
      d_elem = generate_element_vector_analytical(x_i, dx_i, f) 
    elif mode == 1:
      if i ==0: 
        print('Assembling source vector numerically...')
      d_elem =  generate_element_vector_numerical(x_i, dx_i, f)    
    elif mode == 2:
      if i ==0: 
        print('Assembling source vector symbolically...')
      d_elem = generate_element_vector_symbolic(x_i,dx_i,f)
    for j in range(elmat.shape[1]):
      # at vertex i we have contributions phi1*f_i + phi0*f_i
      d[elmat[i,j]] += d_elem[j]
  return d

def plot_solution(x, u, n, D, a, alpha, beta, gamma, mode):
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
   
  mode = 1 # 0: analytical, 1: numerical, 2: symbolic
  
  f = define_source_function(alpha, beta, gamma, mode)
    
  x_vert, x_elem, dx_elem = generate_mesh(n)
  elmat = generate_topology(n)  
    
  s = assemble_stiffness_matrix(n, elmat, x_elem, dx_elem, D, a, mode)
  print(s)
  
  d = assemble_source_vector(n, elmat, x_elem, dx_elem, f, mode)
  print(d)
  
  u = np.linalg.solve(s,d) # solve for solution values at vertices
  print(u)
  
  plot_solution(x_vert, u, n, D, a, alpha, beta, gamma, mode)
      
if __name__=='__main__':
  main()
  

  
  
  