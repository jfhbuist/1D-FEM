#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:50:37 2021

@author: jurriaan
"""

"""
FEM solver 1D

Creates stiffness matrix S and source vector d, to set up the problem
  S*c = d
(with c = u at vertices)

Split domain integral (\int_0^1) into integrals over the elements e_k

Continuous Galerkin: Test functions equal to basis functions
"""

import numpy as np
from scipy.integrate import quad

def reduce_lambda(func, args):
    # this function takes a lambda function of multiple variables, and returns 
    # a lambda function of a single variable
    # args can contain an arbitrary number of arguments
    # asterisk unpacks tuple
    return lambda x: func(x, *args) 


class Grid:
    def __init__(self, L, n):
        self.L = L
        self.n = n
        self.x_vert, self.x_elem, self.dx_elem, self.x_bound, self.loc_bound = self.generate_mesh()
        self.elmat = self.generate_topology()
        self.elbmat = self.generate_boundary_topology()
  
    def generate_mesh(self):
        L = self.L
        n = self.n
        x_vert = np.linspace(0,L,n)
        x_elem = (x_vert[0:n-1]+x_vert[1:n])/2
        #dx = 1/(n-1)
        dx_elem = np.diff(x_vert) # width of each element
        x_bound = np.array([0,L]) # boundary element coordinates
        loc_bound = ["left", "right"] # boundary element locations
        return x_vert, x_elem, dx_elem, x_bound, loc_bound
    
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
            
    def generate_boundary_topology(self):
        n = self.n
        x_bound = self.x_bound
        # This matrix list the vertices of each boundary element. 
        # ie for boundary element i, evaluate elbmat(i), to list the vertex to which it is connected
        elbmat = np.zeros((len(x_bound),1)).astype(int)
        elbmat[0] = 0 # first boundary element is connected to vertex 0
        elbmat[1] = n-1 # last boundary element is connected to vertex n-1
        return elbmat
        
    
class Discretization:
    def __init__(self):
        self.basis_functions = self.define_basis_functions()
    
    def define_basis_functions(self):
        # Define basis functions on element i (actually half of the basis 
        # function associated with a vertex).
        # x_i = middle of element
        # dx_i is width of element
        # x0 = x_i - dx_i/2  # location of left boundary vertex
        # x1 = x_i + dx_i/2  # location of right boundary vertex     
        # for a given element, phi0 is 1 at the left boundary vertex
        # for a given element, phi1 is 1 at the right boundary vertex
        # define as python function
        phi0 = lambda x, x_i, dx_i : (x_i - x)/dx_i + 1/2  # (x1 - x )/dx # corresponds to elmat[i,0]
        phi1 = lambda x, x_i, dx_i : (x - x_i)/dx_i + 1/2  # (x  - x0)/dx # corresponds to elmat[i,1]
        basis_functions = [phi0, phi1]
        return basis_functions 
  
class DiscreteOperator:
    def __init__(self, grid, discretization):
        self.grid = grid
        self.discretization = discretization
   
    def generate_basis_functions(self, x_i, dx_i):
        general_basis_functions = self.discretization.basis_functions  
        local_basis_functions = list()
        for idx, general_basis_function in enumerate(general_basis_functions): 
            local_basis_functions.append(reduce_lambda(general_basis_function, (x_i, dx_i)))
            # reduce_lambda necessary to eliminate elusive bug, where local basis function set in certain iteration, would change in subsequent iteration
        return local_basis_functions
       
    
class Source(DiscreteOperator):
    # f
    # weak form:
    # \int_0^L f*v dx
    def __init__(self, grid, discretization, f):
        super().__init__(grid, discretization)
        self.f = f
        self.d = self.assemble_source_vector()
  
    def generate_integrand(self, test_function_xi, x_i, dx_i):
        f = self.f
        x0 = x_i - dx_i/2
        x1 = x_i + dx_i/2
        f_xi = lambda xi : f(xi*dx_i+x0) # transform f to function of xi
        integrand = lambda xi : f_xi(xi)*test_function_xi(xi)*dx_i
        return integrand
  
    def assemble_source_vector(self):
        # Operates on vertices. For each vertex, sum the contributions of all 
        # its neighbouring elements.
        grid = self.grid
        d = np.zeros(grid.n)
        for i, x_i in enumerate(grid.x_elem): # loop over elements
            # x_i is center of current element
            dx_i = grid.dx_elem[i] # get width of current element
            d_elem =  self.generate_element_vector(x_i, dx_i)    
            for j in range(grid.elmat.shape[1]): # loop over equations for vertex coefficients
                # Each equation is associated with one test function.  
                # We are now considering the contribution of one element to two different equations (in 1D).
                # Each equation is associated with two elements (in 1D).
                # So the equation will be revisted when considering a different element.
                # From vertex i we have contributions (in 1D):
                # \int_{x_{i-1}}^{x_{i}} phi1*f + \int_{x_{i}}^{x_{i+1}} phi0*f 
                d[grid.elmat[i,j]] += d_elem[j]
                # index: equation/test function
        return d
  
    def generate_element_vector(self, x_i, dx_i):
        # Operates on elements.
        x0 = x_i - dx_i/2
        x1 = x_i + dx_i/2
        # get basis functions
        basis_functions = self.generate_basis_functions(x_i, dx_i)
        d_elem = np.zeros(len(basis_functions))
        for j, test_function in enumerate(basis_functions): # loop over test functions
            # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
            test_function_xi = lambda xi : test_function(xi*dx_i+x0) # transform test function to function of xi
            # we integrate over xi, so multiply by dx to get integral over x
            integrand = self.generate_integrand(test_function_xi, x_i, dx_i)
            # integrate over element and put in element vector
            d_elem[j] = quad(integrand,0,1)[0] 
        return d_elem

  
class StiffnessMatrix(DiscreteOperator):
    def __init__(self, grid, discretization, operators):
        super().__init__(grid, discretization)
        for operator in operators:
            operator.s = self.assemble_stiffness_matrix(operator)
        self.s = self.combine_operators(operators)
        
    def combine_operators(self, operators):
        s = np.zeros(operators[0].s.shape)
        for idx, operator in enumerate(operators):
            s = s + operator.s
        return s
    
    def assemble_stiffness_matrix(self, operator):
        # Operates on vertices. For each vertex, sum the contributions of all 
        # its neighbouring elements. Each vertex has an accompanying linear 
        # basis function. In 1D this is composed of phi1 operating on its left 
        # element, and phi0 operating on its right element.
        grid = self.grid
        s = np.zeros((grid.n,grid.n)) # n = number of vertices
        for i, x_i in enumerate(grid.x_elem): # loop over elements
            # x_i is center of current element
            dx_i = grid.dx_elem[i] # get width of current element
            s_elem = self.generate_element_matrix(operator, x_i, dx_i)
            for j in range(grid.elmat.shape[1]): # loop over equations for vertex coefficients
                # Each equation is associated with one test function.  
                # We are now considering the contribution of one element to two different equations (in 1D).
                # Each equation is associated with two elements (in 1D).
                # So the equation will be revisted when considering a different element.
                for k in range(grid.elmat.shape[1]): # loop over solution basis functions 
                    # Each basis function is associated with one vertex.
                    # For each element-equation combination, there are two contributing vertices (in 1D).
                    # Each vertex is associated with two equations and two elements (in 1D).
                    # So the vertex will be revisited three times (in 1D).
                    # From vertex i we have contributions (in 1D):
                    # \int_{x_{i-1}}^{x_{i}} phi0*phi1*c_i + \int_{x_{i-1}}^{x_{i}} phi1*phi1 c_i + \int_{x_{i}}^{x_{i+1}} phi0*phi0 c_i + \int_{x_{i}}^{x_{i+1}} phi1*phi0 c_i    
                    s[grid.elmat[i,j],grid.elmat[i,k]] += s_elem[j,k]
                    # first index: equation/test function, second index: vertex coefficient/basis function
        return s

    def generate_element_matrix(self, operator, x_i, dx_i):
        # Element matrix, operates on elements.  
        # Matrix should be symmetric. Numerical calculation.
        # get vertex coordinates
        x0 = x_i - dx_i/2  # location of left boundary vertex
        x1 = x_i + dx_i/2  # location of right boundary vertex
        # get basis functions
        basis_functions = self.generate_basis_functions(x_i, dx_i) 
        # calculate s_ij
        s_elem = np.zeros((len(basis_functions),len(basis_functions)))
        for j, test_function in enumerate(basis_functions): # loop over test functions
            for k, basis_function in enumerate(basis_functions): # loop over solution basis functions
                # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
                test_function_xi = lambda xi : test_function(xi*dx_i+x0) # transform test function to function of xi
                basis_function_xi = lambda xi : basis_function(xi*dx_i+x0) # transform basis function to function of xi
                integrand = operator.generate_integrand(test_function_xi, basis_function_xi, x_i, dx_i)
                # integrate over element and put in element matrix
                s_elem[j,k] = quad(integrand,0,1)[0] 
        return s_elem    
  
class Diffusion:
    # -D*u_xx
    # weak form:
    # -[D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx    
    def __init__(self, D):
        self.coeff = D
    
    def generate_integrand(self, test_function_xi, basis_function_xi, x_i, dx_i):
        # generate integrand for diffusion
        # calculate numerical derivative of function using central scheme
        ddx = lambda func, x, dx : (func(x+dx/4)-func(x-dx/4))/(dx/2)
        # multiply ddxi by (1/dx) to get ddx (as function of xi)
        # to take the derivative to xi we need the following:
        dxi = dx_i/dx_i # = 1
        # we integrate over xi, so multiply by dx to get integral over x
        integrand = lambda xi : (self.coeff*ddx(test_function_xi,xi,dxi)*(1/dx_i)*ddx(basis_function_xi,xi,dxi)*(1/dx_i))*dx_i
        return integrand
    
    def generate_boundary_integrand(self, bc, x_bound, loc_bound):
        # since we reduce the order of the diffusion operator through integration by parts, boundary terms appear, which must be added to the equation
        # since this term will be added to right-hand side, it gets a minus sign
        if bc[loc_bound][0] == "neumann":
            # in 1D case there is nothing to integrate, test function at boundary is just 1
            integrand = self.coeff*bc[loc_bound][1]*1
        else:
            integrand = 0
        return integrand
                    
    
class Reaction:
    # R*u
    # weak form:
    # \int_0^L R*u*v dx
    def __init__(self, R):
        self.coeff = R
    
    def generate_integrand(self, test_function_xi, basis_function_xi, x_i, dx_i):
        # generate integrand for reaction
        # we integrate over xi, so multiply by dx to get integral over x
        integrand = lambda xi : (self.coeff*test_function_xi(xi)*basis_function_xi(xi))*dx_i
        # integrate over element and put in element matrix
        return integrand
    
    def generate_boundary_integrand(self, bc, x_bound, loc_bound):
        # the boundary terms are zero for the reaction operator, since there is no integration by parts
        integrand = 0
        return integrand
    
    
class Advection:
    # A*u_x
    # weak form:
    # \int_0^L A*(du/dx)*v dx 
    def __init__(self, A):
        self.coeff = A
        
    def generate_integrand(self, test_function_xi, basis_function_xi, x_i, dx_i):
        # generate integrand for linear advection
        # calculate numerical derivative of function using central scheme
        ddx = lambda func, x, dx : (func(x+dx/4)-func(x-dx/4))/(dx/2)
        # multiply ddxi by (1/dx) to get ddx (as function of xi)
        # to take the derivative to xi we need the following:
        dxi = dx_i/dx_i # = 1
        # we integrate over xi, so multiply by dx to get integral over x
        integrand = lambda xi : (self.coeff*ddx(basis_function_xi,xi,dxi)*(1/dx_i)*test_function_xi(xi))*dx_i
        return integrand
        
    def generate_boundary_integrand(self, bc, x_bound, loc_bound):
        # the boundary terms are zero for the advection operator, since there is no integration by parts
        integrand = 0
        return integrand
    

class NaturalBoundary(DiscreteOperator):
    def __init__(self, grid, discretization, operators, bc):
        super().__init__(grid, discretization)
        self.bc = bc
        for operator in operators:
            operator.b_nat = self.assemble_natural_boundary_vector(operator)
        self.b_nat = self.combine_operators(operators)
        
    def combine_operators(self, operators):
        b_nat = np.zeros(self.grid.n)
        for idx, operator in enumerate(operators):
            b_nat = b_nat + operator.b_nat
        return b_nat
        
    def generate_natural_boundary_term(self, operator, boundary_element_idx):
        # Operates on boundary element
        # natural boundary conditions are implicitly satisfied by the formulation
        grid = self.grid
        xb = grid.x_bound[boundary_element_idx]
        lb = grid.loc_bound[boundary_element_idx]
        integrand = operator.generate_boundary_integrand(self.bc, xb, lb) 
        # We need to integrate over the boundary. In 1D this entails flipping the sign of left boundary condition. 
        if lb == 'left':
            bt = - integrand
        elif lb == 'right':
            bt = integrand         
        return bt
    
    def assemble_natural_boundary_vector(self, operator):
        # Operates on vertices
        grid = self.grid
        b = np.zeros(grid.n)
        # loop over boundary elements 
        for i, x_i in enumerate(grid.x_bound): # x_i is center of current boundary element
            bt = self.generate_natural_boundary_term(operator, i)
            # assign contributions from boundary element i to every connected vertex (only 1 in 1D)
            for j in range(grid.elbmat.shape[1]):     
                # each boundary vertex is associated with one test function (in 1D),
                # which is associated with one equation
                b[grid.elbmat[i,j]] += bt
                # grid.elbmat[i,j] is the index of the vertex/test function/equation
        return b
    
    
class Solution(DiscreteOperator):
    def __init__(self, grid, discretization, bc, stiffness, source, natural_boundary, x):
        super().__init__(grid, discretization)
        self.bc = bc
        self.u = self.calculate_solution(stiffness, source, natural_boundary, x)
        
    def calculate_solution(self, stiffness, source, natural_boundary, x):
        grid = self.grid
        discretization = self.discretization
        bc = self.bc
        s = stiffness.s
        d = source.d
        b_nat = natural_boundary.b_nat
        
        # now we handle the essential boundary terms (ie dirichlet boundary conditions)
        
        # theoretical view is to decompose the solution into: u = u_0 + g_tilde 
        # so we solve the following equation for u_0:
        # s*u_0 = d + b_nat - u*g_tilde 
        # then we set up the homegenous dirichlet problem for u_0
        # this includes setting h[idx1] = 0
        # after linear solve the g vector is added back to the solution, to obtain u
        # below, we instead take a (equivalent) practical approach, in which we 
        # just set the values of the boundary nodes to the values of the dirichlet boundary conditions
        # and move their contribution (in the equations for the interior nodes) to the right hand side
        
        # g is a vector containing set values for nodes lying on dirichlet boundary
        g = np.zeros(grid.n) 
        for idx0, xb in enumerate(grid.x_bound): # loop over boundary elements
            lb = grid.loc_bound[idx0]
            for idx1 in grid.elbmat[idx0]: # loop over vertices connected to boundary element
                xv = grid.x_vert[idx1] # position of vertex
                if bc[lb][0] == "dirichlet":
                    # set value for this boundary node
                    g[idx1] = bc[lb][1]
                                                    
        # right-hand side contains contributions from source, natural boundary conditions, and dirichlet boundary conditions             
        h = d + b_nat - np.matmul(s,g)
        # substracting the latter term just means we move terms in the equations for the interior points to the right hand side
        # these are the terms involving boundary points
        
        # conduct same loop as before
        # modify stiffness matrix and rhs vector to implement dirichlet boundary conditions
        for idx0, xb in enumerate(grid.x_bound): # loop over boundary elements
            lb = grid.loc_bound[idx0]
            for idx1 in grid.elbmat[idx0]: # loop over vertices connected to boundary element
                xv = grid.x_vert[idx1] # position of vertex
                if bc[lb][0] == "dirichlet":
                    # eliminate row in stiffness matrix and replace with diagonal 1
                    # this way we get an equation such that 1*boundary_vertex = ...
                    s[idx1] = 0
                    # eliminate column in stiffness matrix
                    # this is possible because we have moved these terms to the right hand side by subtracting np.matmul(s,g) (aka forward substitution)
                    s[:,idx1] = 0
                    s[idx1,idx1] = 1
                    
                    # eliminate row in rhs
                    # this changes the equations for the boundary vertices into 1*boundary_vertex = bc
                    h[idx1] = g[idx1]
                    
        # solve for solution values at vertices
        # these are actually the coefficients associated with the basis 
        # functions centered at each grid point    
        c = np.linalg.solve(s,h) 
        # construct solution at arbitrary locations x, using basis functions
        u = self.construct_solution(grid, discretization, c, x)
        return u        
   
    def construct_solution(self, grid, discretization, c, sol_locs):
        sol = np.zeros(len(sol_locs))
        # loop over solution coordinates
        for idx_sol, sol_loc in enumerate(sol_locs):
            # loop over elements
            for i, x_i in enumerate(grid.x_elem): # x_i is center of current element
                dx_i = grid.dx_elem[i]
                if (sol_loc >= x_i - dx_i) and (sol_loc <= x_i + dx_i):
                    # coordinate is in this element
                    # get basis functions
                    basis_functions = self.generate_basis_functions(x_i, dx_i)                       
                    # loop over bounding vertices
                    for j in range(grid.elmat.shape[1]): 
                        # we assume the basis functions and the rows of the elmat are ordered correspondingly
                        sol[idx_sol] += c[grid.elmat[i,j]]*discretization.basis_functions[j](sol_loc, x_i, dx_i)
                    # due to >= and <= signs in if statement it would be possible to double count this sol_loc
                    # so break this loop and move on to next sol_loc
                    break 
        return sol
                