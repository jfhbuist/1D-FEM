#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import quad
from scipy.misc import derivative

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
        self.dim = dim
        if self.dim == 1:
            n = nx
            self.loc_bound = ["left", "right"]  # boundary element locations
            self.xyz_vert, self.xyz_elem, self.xyz_bound, self.elmat, self.belmat = self.generate_mesh_1D(L, n)
        elif self.dim == 2:
            self.loc_bound = ["left", "right", "bottom", "top"]  # boundary element locations
            self.xyz_vert, self.xyz_elem, self.xyz_bound, self.elmat, self.belmat = self.generate_mesh_2D(L, H, nx, ny)

    def generate_mesh_1D(self, L, n):
        xyz_vert = np.linspace(0, L, n)
        xyz_elem = (xyz_vert[0:n-1]+xyz_vert[1:n])/2
        # dx = 1/(n-1)
        xyz_bound = np.array([0, L])  # boundary element coordinates

        # This matrix lists the vertices of each element.
        # ie for element i, evaluate elmat[i], to list the two vertices which
        # border element i
        elmat = np.zeros((n-1, 2)).astype(int)
        for idx, element in enumerate(elmat):
            elmat[idx, 0] = idx
            elmat[idx, 1] = idx + 1

        # This matrix list the vertices of each boundary element.
        # ie for boundary element i, evaluate belmat(i), to list the vertex to which it is connected
        belmat = np.zeros((len(xyz_bound), 1)).astype(int)
        belmat[0] = 0  # first boundary element is connected to vertex 0
        belmat[1] = n-1  # last boundary element is connected to vertex n-1
        return xyz_vert, xyz_elem, xyz_bound, elmat, belmat

    def generate_mesh_2D(self, L, H, nx, ny):
        # Create vertices with coordinates
        x = np.linspace(0, L, nx)
        y = np.linspace(0, H, ny)
        xm, ym = np.meshgrid(x, y)
        xyz_vert = np.zeros((nx*ny, 2))
        idx = 0
        for i in range(nx):
            for j in range(ny):
                xyz_vert[idx] = np.array([xm[j, i], ym[j, i]])
                idx += 1

        # Make triangle elements, based on squares
        xyz_elem = np.zeros((2*(nx-1)*(ny-1), 2))
        elmat = np.zeros((len(xyz_elem), 3)).astype(int)
        t_idx = 0
        for i in range(nx-1):
            for j in range(ny-1):
                elmat[t_idx, 0] = i*ny+j  # lower left
                elmat[t_idx, 1] = i*ny+ny+j+1  # upper right
                elmat[t_idx, 2] = i*ny+j+1  # upper left
                xyz_elem[t_idx] = (xyz_vert[elmat[t_idx, 0]] + xyz_vert[elmat[t_idx, 1]] + xyz_vert[elmat[t_idx, 2]])/3
                t_idx += 1
                elmat[t_idx, 0] = i*ny+j  # lower left
                elmat[t_idx, 1] = i*ny+ny+j  # lower right
                elmat[t_idx, 2] = i*ny+ny+j+1  # upper right
                xyz_elem[t_idx] = (xyz_vert[elmat[t_idx, 0]] + xyz_vert[elmat[t_idx, 1]] + xyz_vert[elmat[t_idx, 2]])/3
                t_idx += 1
        # plt.scatter(xyz_elem[:,0],xyz_elem[:,1])

        # Create boundary elements with coordinates for their centers
        xyz_bound = np.zeros((2*(nx-1)+2*(ny-1), 2))
        belmat = np.zeros((len(xyz_bound), 2)).astype(int)
        b_idx = 0
        for j in range(ny-1):  # left
            belmat[b_idx, 0] = 0 + j
            belmat[b_idx, 1] = 0 + j + 1
            xyz_bound[b_idx] = (xyz_vert[belmat[b_idx, 0]] + xyz_vert[belmat[b_idx, 1]])/2
            b_idx += 1
        for j in range(ny-1):  # right
            belmat[b_idx, 0] = (nx-1)*ny + j
            belmat[b_idx, 1] = (nx-1)*ny + j + 1
            xyz_bound[b_idx] = (xyz_vert[belmat[b_idx, 0]] + xyz_vert[belmat[b_idx, 1]])/2
            b_idx += 1
        for i in range(nx-1):  # bottom
            belmat[b_idx, 0] = 0 + i*ny
            belmat[b_idx, 1] = 0 + i*ny + ny
            xyz_bound[b_idx] = (xyz_vert[belmat[b_idx, 0]] + xyz_vert[belmat[b_idx, 1]])/2
            b_idx += 1
        for i in range(nx-1):  # top
            belmat[b_idx, 0] = ny-1 + i*ny
            belmat[b_idx, 1] = ny-1 + i*ny + ny
            xyz_bound[b_idx] = (xyz_vert[belmat[b_idx, 0]] + xyz_vert[belmat[b_idx, 1]])/2
            b_idx += 1
        # plt.scatter(xyz_bound[:,0],xyz_bound[:,1])

        return xyz_vert, xyz_elem, xyz_bound, elmat, belmat


class Discretization:
    def __init__(self, dim):
        self.dim = dim
        # self.basis_functions = self.define_basis_functions()

    def generate_basis_functions(self, vert_coords):
        # Generate basis or test functions for current element (fill in vertex coordinates)
        vert_coords_flat = tuple(np.array(vert_coords).reshape(-1))
        general_basis_functions = self.define_basis_functions()
        local_basis_functions = list()
        for idx, general_basis_function in enumerate(general_basis_functions):
            local_basis_functions.append(reduce_lambda(general_basis_function, vert_coords_flat))
            # reduce_lambda necessary to eliminate elusive bug, where local basis function set in certain iteration, would change in subsequent iteration
        return local_basis_functions

    def define_basis_functions(self):
        if self.dim == 1:
            basis_functions = self.define_basis_functions_1D()
        elif self.dim == 2:
            basis_functions = self.define_basis_functions_2D()
        return basis_functions

    def define_basis_functions_1D(self):
        # Define basis functions for an element defined by its two bounding vertices
        # x_i = middle of element
        # dx_i is width of element
        # x0 = x_i - dx_i/2  # location of left boundary vertex
        # x1 = x_i + dx_i/2  # location of right boundary vertex
        # for a given element, phi0 is 1 at the left vertex, and corresponds to elmat[i,0]
        # for a given element, phi1 is 1 at the right vertex, and corresponds to elmat[i,1]
        # define as python function
        phi0 = lambda x, x0, x1: (x1-x)/(x1-x0)
        phi1 = lambda x, x0, x1: (x-x0)/(x1-x0)
        basis_functions = [phi0, phi1]
        return basis_functions

    def define_basis_functions_2D(self):
        # Define basis functions for an element defined by its three bounding vertices
        # These can be oriented arbitrarily, but the canonical element is a right-angled triangle
        # with vertices (0,0), (1,0), (0,1).
        # x0, y0: location of bottom left boundary vertex
        # x1, y1: location of bottom right boundary vertex
        # x2, y2: location of top left boundary vertex
        # for a given element, phi0 is 1 at the bottom left vertex, and corresponds to elmat[i,0]
        # for a given element, phi1 is 1 at the bottom right vertex, and corresponds to elmat[i,1]
        # for a given element, phi2 is 1 at top left vertex, and corresponds to elmat[i,2]
        # calculate area of triangle:
        # 2A = (x0-x1)*(y1-y2)+(y0-y1)*(x2-x1)
        # define as python function
        phi0 = lambda x, y, x0, y0, x1, y1, x2, y2: (1/((x0-x1)*(y1-y2)+(y0-y1)*(x2-x1)))*((y1-y2)*(x-x1)+(x2-x1)*(y-y1))
        phi1 = lambda x, y, x0, y0, x1, y1, x2, y2: (1/((x0-x1)*(y1-y2)+(y0-y1)*(x2-x1)))*((y2-y0)*(x-x2)+(x0-x2)*(y-y2))
        phi2 = lambda x, y, x0, y0, x1, y1, x2, y2: (1/((x0-x1)*(y1-y2)+(y0-y1)*(x2-x1)))*((y0-y1)*(x-x0)+(x1-x0)*(y-y0))
        basis_functions = [phi0, phi1, phi2]
        return basis_functions

    def coordinate_transformation(self, vert_coords, function_x):
        # Transform function of x to function of xi
        if self.dim == 1:
            function_xi = self.coordinate_transformation_1D(vert_coords, function_x)
        elif self.dim == 2:
            function_xi = self.coordinate_transformation_2D(vert_coords, function_x)
        return function_xi

    def coordinate_transformation_1D(self, vert_coords, function_x):
        # Transform 1D function of x to function of xi
        # Use coordinate transformation xi = (x - x0)/dx, dxi/dx = 1/dx, dx = dx dxi
        # This implies x = xi*dx + x0
        x0 = vert_coords[0]
        x1 = vert_coords[1]
        dx_i = abs(x1 - x0)
        function_xi = lambda xi: function_x(xi*dx_i+x0)
        return function_xi

    def coordinate_transformation_inverse(self, vert_coords, function_xi):
        # Transform function of xi to function of x
        if self.dim == 1:
            function_xi = self.coordinate_transformation_inverse_1D(vert_coords, function_xi)
        elif self.dim == 2:
            function_xi = self.coordinate_transformation_inverse_2D(vert_coords, function_xi)
        return function_xi

    def coordinate_transformation_inverse_1D(self, vert_coords, function_xi):
        # Transform 1D function of xi to function of x
        # Use coordinate transformation xi = (x - x0)/dx, dxi/dx = 1/dx, dx = dx dxi
        x0 = vert_coords[0]
        x1 = vert_coords[1]
        dx_i = abs(x1 - x0)
        function_x = lambda x: function_xi((x - x0)/dx_i)
        return function_x

    def integrate_element(self, vert_coords, integrand):
        # Integrate function of xi over element
        if self.dim == 1:
            result = self.integrate_element_1D(vert_coords, integrand)
        elif self.dim == 2:
            result = self.integrate_element_2D(vert_coords, integrand)
        return result

    def integrate_element_1D(self, vert_coords, integrand):
        # Integrate 1D function of xi over element
        x0 = vert_coords[0]
        x1 = vert_coords[1]
        dx_i = abs(x1 - x0)
        # we integrate over xi, so multiply by dx to get integral over x
        result = quad(integrand, 0, 1)[0]*dx_i
        return result

    def differentiate(self, vert_coords, function):
        # Differentiate function of xi to x within element
        if self.dim == 1:
            result = self.differentiate_1D(vert_coords, function)
        elif self.dim == 2:
            result = self.differentiate_2D(vert_coords, function)
        return result

    def differentiate_1D(self, vert_coords, function):
        # Differentiate 1D function of xi to x within element
        x0 = vert_coords[0]
        x1 = vert_coords[1]
        dx_i = abs(x1 - x0)
        # calculate numerical derivative of function using central differences
        result = lambda xi: derivative(function, xi, 1e-6)*(1/dx_i)
        # multiply ddxi by (1/dx) to get ddx (as function of xi)
        return result

    def calculate_element_size(self, vert_coords):
        # Calculate the size of an element
        if self.dim == 1:
            size = self.calculate_element_size_1D(vert_coords)
        elif self.dim == 2:
            size = self.calculate_element_size_2D(vert_coords)
        return size

    def calculate_element_size_1D(self, vert_coords):
        # Calculate length of element
        x0 = vert_coords[0]
        x1 = vert_coords[1]
        dx_i = abs(x1 - x0)
        return dx_i

    def calculate_element_size_2D(self, vert_coords):
        # Calculate area of element
        x0 = vert_coords[0][0]
        y0 = vert_coords[0][1]
        x1 = vert_coords[1][0]
        y1 = vert_coords[1][1]
        x2 = vert_coords[2][0]
        y2 = vert_coords[2][1]
        area = ((x0-x1)*(y1-y2)+(y0-y1)*(x2-x1))/2
        return area

    def check_if_point_in_element(self, vert_coords, point_coords):
        # Check if point with given coordinates is in the element defined by vert_coords
        if self.dim == 1:
            check = self.check_if_point_in_element_1D(vert_coords, point_coords)
        elif self.dim == 2:
            check = self.check_if_point_in_element_2D(vert_coords, point_coords)
        return check

    def check_if_point_in_element_1D(self, vert_coords, point_coords):
        # Check if point with given coordinates is in the 1D element defined by vert_coords
        func_xi = lambda xi: xi
        func_x = self.coordinate_transformation_inverse_1D(vert_coords, func_xi)
        xp = point_coords
        xi = func_x(xp)
        if (xi >= 0) and (xi <= 1):
            check = True
        else:
            check = False
        return check


class SourceOperator:
    # Discrete operator determined by a set function
    def __init__(self, grid, discretization, operators):
        self.grid = grid
        self.discretization = discretization
        self.d = self.combine_operators(operators)

    def combine_operators(self, operators):
        d = np.zeros(len(self.grid.xyz_vert))
        for idx, operator in enumerate(operators):
            d = d + operator.d
        return d

    def assemble_source_vector(self):
        # Operates on vertices. For each vertex, sum the contributions of all
        # its neighbouring elements.
        grid = self.grid
        d = np.zeros(len(grid.xyz_vert))
        for i, x_i in enumerate(grid.xyz_elem):  # loop over elements
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
        discretization = self.discretization
        vert_coords = [grid.xyz_vert[ev] for ev in elem_vertices]
        # get basis functions
        basis_functions = discretization.generate_basis_functions(vert_coords)
        d_elem = np.zeros(len(basis_functions))
        for j, test_function in enumerate(basis_functions):  # loop over test functions
            integrand = self.generate_integrand(test_function, vert_coords)
            # integrate over element and put in element vector
            d_elem[j] = discretization.integrate_element(vert_coords, integrand)
        return d_elem


class Source(SourceOperator):
    def __init__(self, grid, discretization, f):
        self.grid = grid
        self.discretization = discretization
        self.f = f
        self.d = self.assemble_source_vector()

    def generate_integrand(self, test_function, vert_coords):
        # f
        # weak form:
        # \int_0^L f*v dx
        discretization = self.discretization
        f = self.f
        # transform test function to function of xi
        test_function_xi = discretization.coordinate_transformation(vert_coords, test_function)
        # transform f to function of xi
        f_xi = discretization.coordinate_transformation(vert_coords, f)
        integrand = lambda xi: f_xi(xi)*test_function_xi(xi)
        return integrand


class SolutionOperator():
    # Discrete operator acting on the solution
    def __init__(self, grid, discretization, operators):
        self.grid = grid
        self.discretization = discretization
        self.s = self.combine_operators(operators)

    def combine_operators(self, operators):
        s = np.zeros(operators[0].s.shape)
        for idx, operator in enumerate(operators):
            s = s + operator.s
        return s

    def assemble_stiffness_matrix(self):
        # Operates on vertices. For each vertex, sum the contributions of all
        # its neighbouring elements. Each vertex has an accompanying linear
        # basis function. In 1D this is composed of phi1 operating on its left
        # element, and phi0 operating on its right element.
        grid = self.grid
        s = np.zeros((len(grid.xyz_vert), len(grid.xyz_vert)))  # n = number of vertices
        for i, x_i in enumerate(grid.xyz_elem):  # loop over elements
            # x_i is center of current element
            elem_vertices = grid.elmat[i]
            # dx_i = grid.xyz_vert[grid.elmat[i][1]] - grid.xyz_vert[grid.elmat[i][0]]   # get width of current element
            s_elem = self.generate_element_matrix(elem_vertices)
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

    def generate_element_matrix(self, elem_vertices):
        # Element matrix, operates on elements.
        # Matrix should be symmetric. Numerical calculation.
        # get vertex coordinates
        grid = self.grid
        discretization = self.discretization
        vert_coords = [grid.xyz_vert[ev] for ev in elem_vertices]
        # get basis functions
        basis_functions = discretization.generate_basis_functions(vert_coords)
        # calculate s_ij
        s_elem = np.zeros((len(basis_functions), len(basis_functions)))
        for j, test_function in enumerate(basis_functions):  # loop over test functions
            for k, basis_function in enumerate(basis_functions):  # loop over solution basis functions
                integrand = self.generate_integrand(test_function, basis_function, vert_coords)
                # integrate over element and put in element matrix
                s_elem[j, k] = discretization.integrate_element(vert_coords, integrand)
        return s_elem


class NaturalBoundary():
    def __init__(self, grid, discretization, bc, operators):
        self.grid = grid
        self.discretization = discretization
        self.bc = bc
        self.b_nat = self.combine_operators(operators)

    def combine_operators(self, operators):
        b_nat = np.zeros(len(self.grid.xyz_vert))
        for idx, operator in enumerate(operators):
            b_nat = b_nat + operator.b_nat
        return b_nat

    def generate_natural_boundary_term(self, boundary_element_idx):
        # Operates on boundary element
        # natural boundary conditions are implicitly satisfied by the formulation
        grid = self.grid
        bc = self.bc
        xb = grid.xyz_bound[boundary_element_idx]
        lb = grid.loc_bound[boundary_element_idx]
        integrand = self.generate_boundary_integrand(bc, xb, lb)
        # We need to integrate over the boundary. In 1D this entails flipping the sign of left boundary condition.
        if lb == 'left':
            bt = - integrand
        elif lb == 'right':
            bt = integrand
        return bt

    def assemble_natural_boundary_vector(self):
        # Operates on vertices
        grid = self.grid
        b = np.zeros(len(grid.xyz_vert))
        # loop over boundary elements
        for i, x_i in enumerate(grid.xyz_bound):  # x_i is center of current boundary element
            bt = self.generate_natural_boundary_term(i)
            # assign contributions from boundary element i to every connected vertex (only 1 in 1D)
            for j in range(grid.belmat.shape[1]):
                # each boundary vertex is associated with one test function (in 1D),
                # which is associated with one equation
                b[grid.belmat[i, j]] += bt
                # grid.belmat[i,j] is the index of the vertex/test function/equation
        return b


class Diffusion(SolutionOperator, NaturalBoundary):
    # -D*u_xx
    # weak form:
    # -[D*(du/dx)*v]_0^L + \int_0^L D*(du/dx)*(dv/dx) dx
    def __init__(self, grid, discretization, bc, D):
        self.grid = grid
        self.discretization = discretization
        self.bc = bc
        self.coeff = D
        self.s = self.assemble_stiffness_matrix()
        self.b_nat = self.assemble_natural_boundary_vector()

    def generate_integrand(self, test_function, basis_function, vert_coords):
        # generate integrand for diffusion
        discretization = self.discretization
        test_function_xi = discretization.coordinate_transformation(vert_coords, test_function)  # transform test function to function of xi
        basis_function_xi = discretization.coordinate_transformation(vert_coords, basis_function)  # transform basis function to function of xi
        integrand = lambda xi: self.coeff*discretization.differentiate(vert_coords, test_function_xi)(xi)*discretization.differentiate(vert_coords, basis_function_xi)(xi)
        return integrand

    def generate_boundary_integrand(self, bc, xyz_bound, loc_bound):
        # since we reduce the order of the diffusion operator through integration by parts, boundary terms appear, which must be added to the equation
        # since this term will be added to right-hand side, it gets a minus sign
        if bc[loc_bound][0] == "neumann":
            # in 1D case there is nothing to integrate, test function at boundary is just 1
            integrand = self.coeff*bc[loc_bound][1]*1
        else:
            integrand = 0
        return integrand


class Reaction(SolutionOperator, NaturalBoundary):
    # R*u
    # weak form:
    # \int_0^L R*u*v dx
    def __init__(self, grid, discretization, bc, R):
        self.grid = grid
        self.discretization = discretization
        self.bc = bc
        self.coeff = R
        self.s = self.assemble_stiffness_matrix()
        self.b_nat = self.assemble_natural_boundary_vector()

    def generate_integrand(self, test_function, basis_function, vert_coords):
        # generate integrand for reaction
        discretization = self.discretization
        test_function_xi = discretization.coordinate_transformation(vert_coords, test_function)  # transform test function to function of xi
        basis_function_xi = discretization.coordinate_transformation(vert_coords, basis_function)  # transform basis function to function of xi

        integrand = lambda xi: self.coeff*test_function_xi(xi)*basis_function_xi(xi)
        return integrand

    def generate_boundary_integrand(self, bc, xyz_bound, loc_bound):
        # the boundary terms are zero for the reaction operator, since there is no integration by parts
        integrand = 0
        return integrand


class Advection(SolutionOperator, NaturalBoundary):
    # A*u_x
    # weak form:
    # \int_0^L A*(du/dx)*v dx
    def __init__(self, grid, discretization, bc, A):
        self.grid = grid
        self.discretization = discretization
        self.bc = bc
        self.coeff = A
        self.s = self.assemble_stiffness_matrix()
        self.b_nat = self.assemble_natural_boundary_vector()

    def generate_integrand(self, test_function, basis_function, vert_coords):
        # generate integrand for linear advection
        discretization = self.discretization
        test_function_xi = discretization.coordinate_transformation(vert_coords, test_function)  # transform test function to function of xi
        basis_function_xi = discretization.coordinate_transformation(vert_coords, basis_function)  # transform basis function to function of xi

        integrand = lambda xi: self.coeff*discretization.differentiate(vert_coords, basis_function_xi)(xi)*test_function_xi(xi)
        return integrand

    def generate_boundary_integrand(self, bc, xyz_bound, loc_bound):
        # the boundary terms are zero for the advection operator, since there is no integration by parts
        integrand = 0
        return integrand


class Solution():
    def __init__(self, grid, discretization, bc, stiffness, source, natural_boundary, x):
        self.grid = grid
        self.discretization = discretization
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
        g = np.zeros(len(grid.xyz_vert))
        for idx0, xb in enumerate(grid.xyz_bound):  # loop over boundary elements
            lb = grid.loc_bound[idx0]
            for idx1 in grid.belmat[idx0]:  # loop over vertices connected to boundary element
                # xv = grid.xyz_vert[idx1]  # position of vertex
                if bc[lb][0] == "dirichlet":
                    # set value for this boundary node
                    g[idx1] = bc[lb][1]

        # right-hand side contains contributions from source, natural boundary conditions, and dirichlet boundary conditions
        h = d + b_nat - np.matmul(s, g)
        # substracting the latter term just means we move terms in the equations for the interior points to the right hand side
        # these are the terms involving boundary points

        # conduct same loop as before
        # modify stiffness matrix and rhs vector to implement dirichlet boundary conditions
        for idx0, xb in enumerate(grid.xyz_bound):  # loop over boundary elements
            lb = grid.loc_bound[idx0]
            for idx1 in grid.belmat[idx0]:  # loop over vertices connected to boundary element
                # xv = grid.xyz_vert[idx1]  # position of vertex
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
        discretization = self.discretization
        sol = np.zeros(len(sol_locs))
        # loop over solution coordinates
        for idx_sol, sol_loc in enumerate(sol_locs):
            # loop over elements
            for i, x_i in enumerate(grid.xyz_elem):  # x_i is center of current element
                # dx_i = grid.xyz_vert[grid.elmat[i][1]] - grid.xyz_vert[grid.elmat[i][0]]
                elem_vertices = grid.elmat[i]
                vert_coords = [grid.xyz_vert[ev] for ev in elem_vertices]
                if discretization.check_if_point_in_element(vert_coords, sol_loc):
                    # coordinate is in this element
                    # get basis functions
                    basis_functions = discretization.generate_basis_functions(vert_coords)
                    # loop over bounding vertices
                    for j in range(grid.elmat.shape[1]):
                        # we assume the basis functions and the rows of the elmat are ordered correspondingly
                        sol[idx_sol] += c[grid.elmat[i, j]]*basis_functions[j](sol_loc)
                    # due to >= and <= signs in if statement it would be possible to double count this sol_loc
                    # so break this loop and move on to next sol_loc
                    break
        return sol
