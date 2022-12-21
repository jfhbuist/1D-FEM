#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import quad

"""
Created on Thu Nov 18 16:50:37 2021

@author: jurriaan
"""

"""
FEM solver 1D

Creates stiffness matrix S and source vector d, to set up the problem
  S*c = d
(with c = u at vertices)

Split domain integral (\int_0^1) into integrals over the elements e_k  # noqa: W605

Continuous Galerkin: Test functions equal to basis functions
"""


def reduce_lambda(func, args):
    # this function takes a lambda function of multiple variables, and returns
    # a lambda function of a single variable
    # args can contain an arbitrary number of arguments
    # asterisk unpacks tuple
    return lambda x: func(x, *args)


class Grid:
    def __init__(self, dim, L, nx, H=None, ny=None):
        if dim == 1:
            n = nx
            self.loc_bound = ["left", "right"]  # boundary element locations
            self.x_vert, self.x_elem, self.x_bound, self.elmat, self.elbmat = self.generate_mesh_1D(L, n)
        elif dim == 2:
            self.loc_bound = ["left", "right", "bottom", "top"]  # boundary element locations
            self.x_vert, self.x_elem, self.x_bound, self.elmat, self.elbmat = self.generate_mesh_2D(L, H, nx, ny)

    def generate_mesh_1D(self, L, n):
        x_vert = np.linspace(0, L, n)
        x_elem = (x_vert[0:n-1]+x_vert[1:n])/2
        # dx = 1/(n-1)
        x_bound = np.array([0, L])  # boundary element coordinates

        # This matrix lists the vertices of each element.
        # ie for element i, evaluate elmat[i], to list the two vertices which
        # border element i
        elmat = np.zeros((n-1, 2)).astype(int)
        for idx, element in enumerate(elmat):
            elmat[idx, 0] = idx
            elmat[idx, 1] = idx + 1

        # This matrix list the vertices of each boundary element.
        # ie for boundary element i, evaluate elbmat(i), to list the vertex to which it is connected
        elbmat = np.zeros((len(x_bound), 1)).astype(int)
        elbmat[0] = 0  # first boundary element is connected to vertex 0
        elbmat[1] = n-1  # last boundary element is connected to vertex n-1
        return x_vert, x_elem, x_bound, elmat, elbmat

    def generate_mesh_2D(self, L, H, nx, ny):
        # Create vertices with coordinates
        x = np.linspace(0, L, nx)
        y = np.linspace(0, H, ny)
        xm, ym = np.meshgrid(x, y)
        x_vert = np.zeros((nx*ny, 2))
        idx = 0
        for i in range(nx):
            for j in range(ny):
                x_vert[idx] = np.array([xm[j, i], ym[j, i]])
                idx += 1

        # Create square elements with coordinates for their centers
        x_shift = np.linspace(1/(2*(nx-1)), L-1/(2*(nx-1)), nx-1)
        y_shift = np.linspace(1/(2*(ny-1)), H-1/(2*(ny-1)), ny-1)
        xm_shift, ym_shift = np.meshgrid(x_shift, y_shift)
        x_elem_square = np.zeros(((nx-1)*(ny-1), 2))
        idx = 0
        for i in range(nx-1):
            for j in range(ny-1):
                x_elem_square[idx] = np.array([xm_shift[j, i], ym_shift[j, i]])
                idx += 1

        # Make triangle elements, based on squares
        # x_elem = np.zeros((2*len(x_elem_square), 2))
        # elmat = np.zeros((2*len(x_elem_square), 3)).astype(int)
        # for s_idx, s_elem in enumerate(x_elem_square):
        #     t_idx = 2*s_idx
        #     elmat[t_idx, 0] = s_idx  # lower left
        #     elmat[t_idx, 1] = s_idx + nx + 2  # upper right
        #     elmat[t_idx, 2] = s_idx + 1  # upper left
        #     x_elem[t_idx] = (x_vert[elmat[t_idx, 0]] + x_vert[elmat[t_idx, 1]] + x_vert[elmat[t_idx, 2]])/3
        #     t_idx += 1
        #     elmat[t_idx, 0] = s_idx  # lower left
        #     elmat[t_idx, 1] = s_idx + nx + 1  # lower right
        #     elmat[t_idx, 2] = s_idx + nx + 2  # upper right
        #     x_elem[t_idx] = (x_vert[elmat[t_idx, 0]] + x_vert[elmat[t_idx, 1]] + x_vert[elmat[t_idx, 2]])/3
        #     a=1

        # Make triangle elements, based on squares
        x_elem = np.zeros((2*(nx-1)*(ny-1), 2))
        elmat = np.zeros((len(x_elem), 3)).astype(int)
        t_idx = 0
        for i in range(nx-1):
            for j in range(ny-1):
                elmat[t_idx, 0] = i*ny+j  # lower left
                elmat[t_idx, 1] = i*ny+ny+j+1  # upper right
                elmat[t_idx, 2] = i*ny+j+1  # upper left
                x_elem[t_idx] = (x_vert[elmat[t_idx, 0]] + x_vert[elmat[t_idx, 1]] + x_vert[elmat[t_idx, 2]])/3
                t_idx += 1
                elmat[t_idx, 0] = i*ny+j  # lower left
                elmat[t_idx, 1] = i*ny+ny+j  # lower right
                elmat[t_idx, 2] = i*ny+ny+j+1  # upper right
                x_elem[t_idx] = (x_vert[elmat[t_idx, 0]] + x_vert[elmat[t_idx, 1]] + x_vert[elmat[t_idx, 2]])/3
                t_idx += 1
        # plt.scatter(x_elem[:,0],x_elem[:,1])

        # Create boundary elements with coordinates for their centers
        # x_bound = np.zeros((2*(nx-1)+2*(ny-1), 2))
        # idx = 0
        # for j in range(ny-1):  # left
        #     x_bound[idx] = np.array([0, y_shift[j]])
        #     idx += 1
        # for j in range(ny-1):  # right
        #     x_bound[idx] = np.array([L, y_shift[j]])
        #     idx += 1
        # for i in range(nx-1):  # bottom
        #     x_bound[idx] = np.array([x_shift[i], 0])
        #     idx += 1
        # for i in range(nx-1):  # top
        #     x_bound[idx] = np.array([x_shift[i], H])
        #     idx += 1

        # Create boundary elements with coordinates for their centers
        x_bound = np.zeros((2*(nx-1)+2*(ny-1), 2))
        elbmat = np.zeros((len(x_bound), 2)).astype(int)
        b_idx = 0
        for j in range(ny-1):  # left
            elbmat[b_idx, 0] = 0 + j
            elbmat[b_idx, 1] = 0 + j + 1
            x_bound[b_idx] = (x_vert[elbmat[b_idx, 0]] + x_vert[elbmat[b_idx, 1]])/2
            b_idx += 1
        for j in range(ny-1):  # right
            elbmat[b_idx, 0] = (nx-1)*ny + j
            elbmat[b_idx, 1] = (nx-1)*ny + j + 1
            x_bound[b_idx] = (x_vert[elbmat[b_idx, 0]] + x_vert[elbmat[b_idx, 1]])/2
            b_idx += 1
        for i in range(nx-1):  # bottom
            elbmat[b_idx, 0] = 0 + i*ny
            elbmat[b_idx, 1] = 0 + i*ny + ny
            x_bound[b_idx] = (x_vert[elbmat[b_idx, 0]] + x_vert[elbmat[b_idx, 1]])/2
            b_idx += 1
        for i in range(nx-1):  # top
            elbmat[b_idx, 0] = ny-1 + i*ny
            elbmat[b_idx, 1] = ny-1 + i*ny + ny
            x_bound[b_idx] = (x_vert[elbmat[b_idx, 0]] + x_vert[elbmat[b_idx, 1]])/2
            b_idx += 1
        # plt.scatter(x_bound[:,0],x_bound[:,1])

        return x_vert, x_elem, x_bound, elmat, elbmat


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
        phi0 = lambda x, x0, x1: (x1-x)/abs(x1-x0)  # corresponds to elmat[i,0]
        phi1 = lambda x, x0, x1: (x-x0)/abs(x1-x0)  # corresponds to elmat[i,1]
        basis_functions = [phi0, phi1]
        return basis_functions


class DiscreteOperator:
    def __init__(self, grid, discretization):
        self.grid = grid
        self.discretization = discretization

    def generate_basis_functions(self, vert_coords):
        x0 = vert_coords[0]
        x1 = vert_coords[1]
        dx_i = abs(x1 - x0)
        general_basis_functions = self.discretization.basis_functions
        local_basis_functions = list()
        for idx, general_basis_function in enumerate(general_basis_functions):
            local_basis_functions.append(reduce_lambda(general_basis_function, (x0, x1)))
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

    def generate_integrand(self, test_function, vert_coords):
        x0 = vert_coords[0]
        x1 = vert_coords[1]
        dx_i = abs(x1 - x0)
        f = self.f
        # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
        # transform test function to function of xi
        test_function_xi = lambda xi: test_function(xi*dx_i+x0)
        # transform f to function of xi
        f_xi = lambda xi: f(xi*dx_i+x0)
        # we integrate over xi, so multiply by dx to get integral over x
        integrand = lambda xi: f_xi(xi)*test_function_xi(xi)*dx_i
        return integrand

    def assemble_source_vector(self):
        # Operates on vertices. For each vertex, sum the contributions of all
        # its neighbouring elements.
        grid = self.grid
        d = np.zeros(len(grid.x_vert))
        for i, x_i in enumerate(grid.x_elem):  # loop over elements
            # x_i is center of current element
            elem_vertices = grid.elmat[i]
            d_elem = self.generate_element_vector(elem_vertices)
            for j in range(len(elem_vertices)):  # loop over equations for vertex coefficients
                # Each equation is associated with one test function.
                # We are now considering the contribution of one element to two different equations (in 1D).
                # Each equation is associated with two elements (in 1D).
                # So the equation will be revisted when considering a different element.
                # From vertex i we have contributions (in 1D):
                # \int_{x_{i-1}}^{x_{i}} phi1*f + \int_{x_{i}}^{x_{i+1}} phi0*f
                d[grid.elmat[i, j]] += d_elem[j]
                # index: equation/test function
        return d

    def generate_element_vector(self, elem_vertices):
        # Operates on elements.
        grid = self.grid
        vert_coords = [grid.x_vert[ev] for ev in elem_vertices]
        # get basis functions
        basis_functions = self.generate_basis_functions(vert_coords)
        d_elem = np.zeros(len(basis_functions))
        for j, test_function in enumerate(basis_functions):  # loop over test functions
            integrand = self.generate_integrand(test_function, vert_coords)
            # integrate over element and put in element vector
            d_elem[j] = quad(integrand, 0, 1)[0]
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
        s = np.zeros((len(grid.x_vert), len(grid.x_vert)))  # n = number of vertices
        for i, x_i in enumerate(grid.x_elem):  # loop over elements
            # x_i is center of current element
            elem_vertices = grid.elmat[i]
            # dx_i = grid.x_vert[grid.elmat[i][1]] - grid.x_vert[grid.elmat[i][0]]   # get width of current element
            s_elem = self.generate_element_matrix(operator, elem_vertices)
            for j in range(len(elem_vertices)):  # loop over equations for vertex coefficients
                # Each equation is associated with one test function.
                # We are now considering the contribution of one element to two different equations (in 1D).
                # Each equation is associated with two elements (in 1D).
                # So the equation will be revisted when considering a different element.
                for k in range(len(elem_vertices)):  # loop over solution basis functions
                    # Each basis function is associated with one vertex.
                    # For each element-equation combination, there are two contributing vertices (in 1D).
                    # Each vertex is associated with two equations and two elements (in 1D).
                    # So the vertex will be revisited three times (in 1D).
                    # From vertex i we have contributions (in 1D):
                    # \int_{x_{i-1}}^{x_{i}} phi0*phi1*c_i + \int_{x_{i-1}}^{x_{i}} phi1*phi1 c_i + \int_{x_{i}}^{x_{i+1}} phi0*phi0 c_i + \int_{x_{i}}^{x_{i+1}} phi1*phi0 c_i
                    s[grid.elmat[i, j], grid.elmat[i, k]] += s_elem[j, k]
                    # first index: equation/test function, second index: vertex coefficient/basis function
        return s

    def generate_element_matrix(self, operator, elem_vertices):
        # Element matrix, operates on elements.
        # Matrix should be symmetric. Numerical calculation.
        # get vertex coordinates
        grid = self.grid
        vert_coords = [grid.x_vert[ev] for ev in elem_vertices]
        # get basis functions
        basis_functions = self.generate_basis_functions(vert_coords)
        # calculate s_ij
        s_elem = np.zeros((len(basis_functions), len(basis_functions)))
        for j, test_function in enumerate(basis_functions):  # loop over test functions
            for k, basis_function in enumerate(basis_functions):  # loop over solution basis functions
                integrand = operator.generate_integrand(test_function, basis_function, vert_coords)
                # integrate over element and put in element matrix
                s_elem[j, k] = quad(integrand, 0, 1)[0]
        return s_elem


class Diffusion:
    # -D*u_xx
    # weak form:
    # -[D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx
    def __init__(self, D):
        self.coeff = D

    def generate_integrand(self, test_function, basis_function, vert_coords):
        # generate integrand for diffusion
        x0 = vert_coords[0]
        x1 = vert_coords[1]
        dx_i = abs(x1 - x0)
        # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
        test_function_xi = lambda xi: test_function(xi*dx_i+x0)  # transform test function to function of xi
        basis_function_xi = lambda xi: basis_function(xi*dx_i+x0)  # transform basis function to function of xi

        # calculate numerical derivative of function using central scheme
        ddx = lambda func, x, dx: (func(x+dx/4)-func(x-dx/4))/(dx/2)
        # multiply ddxi by (1/dx) to get ddx (as function of xi)
        # to take the derivative to xi we need the following:
        dxi = dx_i/dx_i  # = 1
        # we integrate over xi, so multiply by dx to get integral over x
        integrand = lambda xi: (self.coeff*ddx(test_function_xi, xi, dxi)*(1/dx_i)*ddx(basis_function_xi, xi, dxi)*(1/dx_i))*dx_i
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

    def generate_integrand(self, test_function, basis_function, vert_coords):
        # generate integrand for reaction
        x0 = vert_coords[0]
        x1 = vert_coords[1]
        dx_i = abs(x1 - x0)
        # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
        test_function_xi = lambda xi: test_function(xi*dx_i+x0)  # transform test function to function of xi
        basis_function_xi = lambda xi: basis_function(xi*dx_i+x0)  # transform basis function to function of xi

        # we integrate over xi, so multiply by dx to get integral over x
        integrand = lambda xi: (self.coeff*test_function_xi(xi)*basis_function_xi(xi))*dx_i
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

    def generate_integrand(self, test_function, basis_function, vert_coords):
        # generate integrand for linear advection
        x0 = vert_coords[0]
        x1 = vert_coords[1]
        dx_i = abs(x1 - x0)
        # Use coordinate transformation xi = (x - x_i)/dx, dxi/dx = 1/dx, dx = dx dxi
        test_function_xi = lambda xi: test_function(xi*dx_i+x0)  # transform test function to function of xi
        basis_function_xi = lambda xi: basis_function(xi*dx_i+x0)  # transform basis function to function of xi

        # calculate numerical derivative of function using central scheme
        ddx = lambda func, x, dx: (func(x+dx/4)-func(x-dx/4))/(dx/2)
        # multiply ddxi by (1/dx) to get ddx (as function of xi)
        # to take the derivative to xi we need the following:
        dxi = dx_i/dx_i  # = 1
        # we integrate over xi, so multiply by dx to get integral over x
        integrand = lambda xi: (self.coeff*ddx(basis_function_xi, xi, dxi)*(1/dx_i)*test_function_xi(xi))*dx_i
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
        b_nat = np.zeros(len(self.grid.x_vert))
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
        b = np.zeros(len(grid.x_vert))
        # loop over boundary elements
        for i, x_i in enumerate(grid.x_bound):  # x_i is center of current boundary element
            bt = self.generate_natural_boundary_term(operator, i)
            # assign contributions from boundary element i to every connected vertex (only 1 in 1D)
            for j in range(grid.elbmat.shape[1]):
                # each boundary vertex is associated with one test function (in 1D),
                # which is associated with one equation
                b[grid.elbmat[i, j]] += bt
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
        g = np.zeros(len(grid.x_vert))
        for idx0, xb in enumerate(grid.x_bound):  # loop over boundary elements
            lb = grid.loc_bound[idx0]
            for idx1 in grid.elbmat[idx0]:  # loop over vertices connected to boundary element
                # xv = grid.x_vert[idx1]  # position of vertex
                if bc[lb][0] == "dirichlet":
                    # set value for this boundary node
                    g[idx1] = bc[lb][1]

        # right-hand side contains contributions from source, natural boundary conditions, and dirichlet boundary conditions
        h = d + b_nat - np.matmul(s, g)
        # substracting the latter term just means we move terms in the equations for the interior points to the right hand side
        # these are the terms involving boundary points

        # conduct same loop as before
        # modify stiffness matrix and rhs vector to implement dirichlet boundary conditions
        for idx0, xb in enumerate(grid.x_bound):  # loop over boundary elements
            lb = grid.loc_bound[idx0]
            for idx1 in grid.elbmat[idx0]:  # loop over vertices connected to boundary element
                # xv = grid.x_vert[idx1]  # position of vertex
                if bc[lb][0] == "dirichlet":
                    # eliminate row in stiffness matrix and replace with diagonal 1
                    # this way we get an equation such that 1*boundary_vertex = ...
                    s[idx1] = 0
                    # eliminate column in stiffness matrix
                    # this is possible because we have moved these terms to the right hand side by subtracting np.matmul(s,g) (aka forward substitution)
                    s[:, idx1] = 0
                    s[idx1, idx1] = 1

                    # eliminate row in rhs
                    # this changes the equations for the boundary vertices into 1*boundary_vertex = bc
                    h[idx1] = g[idx1]

        # solve for solution values at vertices
        # these are actually the coefficients associated with the basis
        # functions centered at each grid point
        c = np.linalg.solve(s, h)
        # construct solution at arbitrary locations x, using basis functions
        u = self.construct_solution(grid, discretization, c, x)
        return u

    def construct_solution(self, grid, discretization, c, sol_locs):
        sol = np.zeros(len(sol_locs))
        # loop over solution coordinates
        for idx_sol, sol_loc in enumerate(sol_locs):
            # loop over elements
            for i, x_i in enumerate(grid.x_elem):  # x_i is center of current element
                # dx_i = grid.x_vert[grid.elmat[i][1]] - grid.x_vert[grid.elmat[i][0]]
                elem_vertices = grid.elmat[i]
                vert_coords = [grid.x_vert[ev] for ev in elem_vertices]
                x0 = vert_coords[0]
                x1 = vert_coords[1]
                if (sol_loc >= x0) and (sol_loc <= x1):
                    # coordinate is in this element
                    # get basis functions
                    # basis_functions = self.generate_basis_functions(x_i, dx_i)
                    # loop over bounding vertices
                    for j in range(grid.elmat.shape[1]):
                        # we assume the basis functions and the rows of the elmat are ordered correspondingly
                        sol[idx_sol] += c[grid.elmat[i, j]]*discretization.basis_functions[j](sol_loc, x0, x1)
                    # due to >= and <= signs in if statement it would be possible to double count this sol_loc
                    # so break this loop and move on to next sol_loc
                    break
        return sol
