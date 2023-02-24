# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:23:49 2022

@author: jfhbuist
"""

import numpy as np
import sympy as sp


def broadcast(fun):
    # this is needed when using lambdify,
    # to make a constant function return a vector when the input is a vector
    return lambda *x: np.broadcast_arrays(fun(*x), *x)[0]


class ExactSolution:
    def get_solution(self, pde, bc_types, bc_params, grid_params, core_params, source_params):
        if pde == 'steady_diffusion_reaction_1D':
            dim = 1
            u_sym, x_sym = self.steady_diffusion_reaction_1D(dim, bc_types, bc_params, grid_params,
                                                             core_params, source_params)
        elif pde == 'steady_advection_diffusion_reaction_1D':
            dim = 1
            u_sym, x_sym = self.steady_advection_diffusion_reaction_1D(dim, bc_types, bc_params, grid_params,
                                                                       core_params, source_params)
        elif pde == 'steady_advection_diffusion_1D':
            dim = 1
            u_sym, x_sym = self.steady_advection_diffusion_1D(dim, bc_types, bc_params, grid_params,
                                                              core_params, source_params)
        elif pde == 'laplace_1D':
            dim = 1
            u_sym, x_sym = self.laplace_1D(dim, bc_types, bc_params, grid_params,
                                           core_params, source_params)
        elif pde == 'laplace_2D':
            dim = 2
            u_sym, x_sym, y_sym = self.laplace_2D(dim, bc_types, bc_params, grid_params,
                                                  core_params, source_params)
        if dim == 1:
            L = grid_params["L"]
            n = grid_params["n"]
            # convert to a python function:
            u_num = broadcast(sp.lambdify(x_sym, u_sym, "numpy"))
            # specify points at which to return function:
            x = np.linspace(0, L, n)
            # evalute function at specified points:
            u = u_num(x)
            return u, x
        elif dim == 2:
            L = grid_params["L"]
            H = grid_params["H"]
            nx = grid_params["nx"]
            ny = grid_params["ny"]
            # convert to a python function:
            u_num = broadcast(sp.lambdify([x_sym, y_sym], u_sym, "numpy"))
            # specify points at which to return function:
            x = np.linspace(0, L, nx)
            y = np.linspace(0, H, ny)
            X, Y = np.meshgrid(x, y)
            # evaluate function at specified points:
            U = u_num(X, Y)
            return U, X, Y

    def steady_diffusion_reaction_1D(self, dim, bc_types, bc_params, grid_params, core_params, source_params):
        """Diffusion-reaction equation (aka Helmholtz equation): -D*u_xx + R*u = f"""
        D = core_params["D"]
        R = core_params["R"]
        L = grid_params["L"]
        source_function = source_params["function"]
        alpha = source_params["alpha"]
        beta = source_params["beta"]
        gamma = source_params["gamma"]

        # see chapter 6, Riley & Hobson (2011)
        # x is our symbolic spatial variable:
        x = sp.symbols("x")

        # complementary solution (general solution to the homogeneous equation):
        # uc = c0*exp(mu_0*x) + c1*exp(mu_1*x)
        mu_0 = np.sqrt(R/D)
        mu_1 = -np.sqrt(R/D)

        if source_function == "periodic":
            # periodic source term:
            # f = alpha + beta*sin(gamma*x)
            # particular solution (this may be any solution to the full inhomogeneous equation,
            # and is source dependent):
            up = alpha/R + beta*sp.sin(gamma*x)/(R + D*gamma**2)

        dupdx = sp.diff(up, x)  # needed for neumann boundary conditions
        # general solution is the sum of the two:
        # u = uc + up
        # constants are set afterwards through boundary conditions

        c0, c1 = self.solve_for_bc_1D(bc_types, bc_params, L, mu_0, mu_1, x, up, dupdx)
        uc = c0*sp.exp(mu_0*x) + c1*sp.exp(mu_1*x)
        u = uc + up  # this is a symbolic function
        return u, x

    def steady_advection_diffusion_reaction_1D(self, dim, bc_types, bc_params, grid_params,
                                               core_params, source_params):
        """Advection-diffusion-reaction equation: A*u_x - D*u_xx + R*u = f"""
        A = core_params["A"]
        D = core_params["D"]
        R = core_params["R"]
        L = grid_params["L"]
        source_function = source_params["function"]
        alpha = source_params["alpha"]
        beta = source_params["beta"]
        gamma = source_params["gamma"]

        # see chapter 6, Riley & Hobson (2011)
        # x is our symbolic spatial variable:
        x = sp.symbols("x")

        # complementary solution (general solution to the homogeneous equation):
        # uc = c0*exp(mu_0*x) + c1*exp(mu_1*x)
        mu_0 = (-A + np.sqrt(A**2 + 4*D*R))/(-2*D)
        mu_1 = (-A - np.sqrt(A**2 + 4*D*R))/(-2*D)

        if source_function == "periodic":
            # periodic source term:
            # f = alpha + beta*sin(gamma*x)
            # particular solution (this may be any solution to the full inhomogeneous equation,
            # and is source dependent):
            b0 = alpha/R
            b1 = beta/(R + D*gamma**2) - beta*((A*gamma)/(R + D*gamma**2)) \
                * ((A*gamma)/((A*gamma)**2 + (R + D*gamma**2)**2))
            b2 = -beta*(A*gamma)/(((A*gamma)**2) + (R + D*gamma**2)**2)
            up = b0 + b1*sp.sin(gamma*x) + b2*sp.cos(gamma*x)

        dupdx = sp.diff(up, x)  # needed for neumann boundary conditions
        # general solution is the sum of the two:
        # u = uc + up
        # constants are set afterwards through boundary conditions

        c0, c1 = self.solve_for_bc_1D(bc_types, bc_params, L, mu_0, mu_1, x, up, dupdx)
        uc = c0*sp.exp(mu_0*x) + c1*sp.exp(mu_1*x)
        u = uc + up  # this is a symbolic function
        return u, x

    def steady_advection_diffusion_1D(self, dim, bc_types, bc_params, grid_params,
                                      core_params, source_params):
        """Advection-diffusion equation: A*u_x - D*u_xx = f"""
        A = core_params["A"]
        D = core_params["D"]
        L = grid_params["L"]
        source_function = source_params["function"]
        alpha = source_params["alpha"]
        beta = source_params["beta"]
        gamma = source_params["gamma"]

        # see chapter 6, Riley & Hobson (2011)
        # x is our symbolic spatial variable:
        x = sp.symbols("x")

        # complementary solution (general solution to the homogeneous equation):
        # uc = c0*exp(mu_0*x) + c1*exp(mu_1*x)
        mu_0 = 0
        mu_1 = A/D

        if source_function == "periodic":
            # periodic source term:
            # f = alpha + beta*sin(gamma*x)
            # particular solution (this may be any solution to the full inhomogeneous equation,
            # and is source dependent):
            b0 = 0
            b1 = beta/(D*gamma**2) - beta*((A*gamma)/(D*gamma**2)) \
                * ((A*gamma)/((A*gamma)**2 + (D*gamma**2)**2))
            b2 = -beta*(A*gamma)/((A*gamma)**2 + (D*gamma**2)**2)
            b3 = alpha/A
            up = b0 + b1*sp.sin(gamma*x) + b2*sp.cos(gamma*x) + b3*x

        dupdx = sp.diff(up, x)  # needed for neumann boundary conditions
        # general solution is the sum of the two:
        # u = uc + up
        # constants are set afterwards through boundary conditions

        c0, c1 = self.solve_for_bc_1D(bc_types, bc_params, L, mu_0, mu_1, x, up, dupdx)
        uc = c0*sp.exp(mu_0*x) + c1*sp.exp(mu_1*x)
        u = uc + up  # this is a symbolic function
        return u, x

    def solve_for_bc_1D(self, bc_types, bc_params, L, mu_0, mu_1, x, up, dupdx):
        """Solve for c0 and c1 using the boundary conditions.

        We will get an equation of the form A*c = B, with c = [c0; c1].
        dirichlet bc: set value of solution at boundary
        neumann bc: set value of normal gradient of solution at boundary
        """
        A = np.zeros((2, 2))
        B = np.zeros(2)
        if bc_types["left"] == "neumann":
            # -(mu_0*c0 + mu_1*c1 + dupdx(0)) = bc(left)
            A[0, 0] = -mu_0
            A[0, 1] = -mu_1
            B[0] = dupdx.subs(x, 0) + bc_params["left"][1]
        elif bc_types["left"] == "dirichlet":
            # c0 + c1 + up(0) = bc(left)
            A[0, 0] = 1
            A[0, 1] = 1
            B[0] = -up.subs(x, 0) + bc_params["left"][1]
        if bc_types["right"] == "neumann":
            # mu_0*c0*exp(mu_0*L) + mu_1*c1*exp(mu_1*L) + dupdx(1) = bc(right)
            A[1, 0] = mu_0*np.exp(mu_0*L)
            A[1, 1] = mu_1*np.exp(mu_1*L)
            B[1] = -dupdx.subs(x, L) + bc_params["right"][1]
        elif bc_types["right"] == "dirichlet":
            # c0*exp(mu_0*L) + c1*exp(mu_1*L) + up(L) = bc(right)
            A[1, 0] = np.exp(mu_0*L)
            A[1, 1] = np.exp(mu_1*L)
            B[1] = -up.subs(x, L) + bc_params["right"][1]
        c = np.linalg.solve(A, B)
        c0 = c[0]
        c1 = c[1]
        return c0, c1

    def laplace_1D(self, dim, bc_types, bc_params, grid_params, core_params, source_params):
        """Laplace equation: -D*u_xx = 0"""
        # D = core_params["D"]
        L = grid_params["L"]

        # see chapter 6, Riley & Hobson (2011)
        # x is our symbolic spatial variable:
        x = sp.symbols("x")

        # complementary solution (general solution to the homogeneous equation):
        # uc = c0 + c1*x

        # no source terms, so no particular solution
        # u = uc
        # constants are set through boundary conditions

        # dirichlet bc: set value of solution at boundary
        # neumann bc: set value of normal gradient of solution at boundary

        A = np.zeros((2, 2))
        B = np.zeros(2)
        if bc_types["left"] == "neumann":
            # -c1 = bc(left)
            A[0, 0] = 0
            A[0, 1] = -1
            B[0] = bc_params["left"][1]
        elif bc_types["left"] == "dirichlet":
            # c0 = bc(left)
            A[0, 0] = 1
            A[0, 1] = 0
            B[0] = bc_params["left"][1]
        if bc_types["right"] == "neumann":
            # c1 = bc(right)
            A[1, 0] = 0
            A[1, 1] = 1
            B[1] = bc_params["right"][1]
        elif bc_types["right"] == "dirichlet":
            # c0 + c1*L = bc(right)
            A[1, 0] = 1
            A[1, 1] = L
            B[1] = bc_params["right"][1]
        c = np.linalg.solve(A, B)
        c0 = c[0]
        c1 = c[1]

        uc = c0 + c1*x
        u = uc
        return u, x

    def laplace_2D(self, dim, bc_types, bc_params, grid_params, core_params, source_params):
        """Laplace equation: -D*(u_xx + u_yy) = 0"""
        # D = core_params["D"]
        L = grid_params["L"]
        H = grid_params["H"]

        bc_func_L = bc_params["left"][0]
        aL = bc_params["left"][1]
        bL = bc_params["left"][2]
        cL = bc_params["left"][3]
        dL = bc_params["left"][4]
        eL = bc_params["left"][5]

        bc_func_R = bc_params["right"][0]
        aR = bc_params["right"][1]
        bR = bc_params["right"][2]
        cR = bc_params["right"][3]
        dR = bc_params["right"][4]
        eR = bc_params["right"][5]

        bc_func_B = bc_params["bottom"][0]
        aB = bc_params["bottom"][1]
        bB = bc_params["bottom"][2]
        cB = bc_params["bottom"][3]
        dB = bc_params["bottom"][4]
        eB = bc_params["bottom"][5]

        bc_func_T = bc_params["top"][0]
        aT = bc_params["top"][1]
        bT = bc_params["top"][2]
        cT = bc_params["top"][3]
        dT = bc_params["top"][4]
        eT = bc_params["top"][5]

        # see chapter 11, Riley & Hobson (2011)
        # and https://tutorial.math.lamar.edu/Classes/DE/LaplacesEqn.aspx
        # x and y are our symbolic spatial variables:
        x = sp.symbols("x")
        y = sp.symbols("y")

        # separation of variables:
        # general separated solution:
        # u = X(x)*Y(y)

        # we solve the boundary value problem in 4 parts
        # Here we give the solution for the problem with four dirichlet boundaries

        # for the left boundary, we take
        # X'' = lambda^2*X, Y'' = -lambda^2*Y
        # X(x) = A*cosh(lambda*(x-L)) + B*sinh(lambda*(x-L))
        # Y(y) = C*cos(lambda*y) + D*sin(lambda*y)
        # lambda_n = n*pi/H
        # Assume homogeneous dirichlet bc everywhere, except at left boundary u = gL(y)
        # A = 0, C = 0
        # u_L = \sum_n c_L_n*sinh(lambda_n*(x-L))*sin(lambda_n*y)
        # c_L_n = (2/H)*(1/sinh(lambda_n*(-L)))*\int_0^H gL(y)*sin(lambda_n*y) dy

        # for the right boundary, we take
        # X'' = lambda^2*X, Y'' = -lambda^2*Y
        # X(x) = A*cosh(lambda*x) + B*sinh(lambda*x)
        # Y(y) = C*cos(lambda*y) + D*sin(lambda*y)
        # lambda_n = n*pi/H
        # Assume homogeneous dirichlet bc everywhere, except at left boundary u = gR(y)
        # A = 0, C = 0
        # u_R = \sum_n c_R_n*sinh(lambda_n*x)*sin(lambda_n*y)
        # c_R_n = (2/H)*(1/sinh(lambda_n*L))*\int_0^H gR(y)*sin(lambda_n*y) dy

        # for the bottom boundary, we take
        # X'' = -lambda^2*X, Y'' = lambda^2*Y
        # X(x) = A*cos(mu*x) + B*sin(mu*x)
        # Y(y) = C*cosh(mu*(y-H)) + D*sinh(mu*(y-H))
        # mu_n = n*pi/L
        # Assume homogeneous dirichlet bc everywhere, except at bottom boundary u = gB(x0)
        # A = 0, C = 0
        # u_B = \sum_n c_B_n*sinh(mu_n*(y-H))*sin(mu_n*x)
        # c_B_n = (2/L)*(1/sinh(mu_n*(-H))*\int_0^L gB(x)*sin(mu_n*x) dx

        # for the top boundary, we take
        # X'' = -lambda^2*X, Y'' = lambda^2*Y
        # X(x) = A*cos(mu*x) + B*sin(mu*x)
        # Y(y) = C*cosh(mu*y) + D*sinh(mu*y)
        # mu_n = n*pi/L
        # Assume homogeneous dirichlet bc everywhere, except at top boundary u = gT(x)
        # A = 0, C = 0
        # u_T = \sum_n c_T_n*sinh(mu_n*x)*sin(mu_n*x)
        # c_T_n = (2/L)*(1/sinh(mu_n*H)*\int_0^L gT(x)*sin(mu_n*x) dx

        if (bc_types["left"] == "dirichlet") and (bc_types["right"] == "dirichlet") and (bc_types["bottom"] == "dirichlet") and (bc_types["top"] == "dirichlet"):
            bc = 1
        elif (bc_types["left"] == "neumann") and (bc_types["right"] == "dirichlet") and (bc_types["bottom"] == "dirichlet") and (bc_types["top"] == "dirichlet"):
            bc = 2
        elif (bc_types["left"] == "dirichlet") and (bc_types["right"] == "neumann") and (bc_types["bottom"] == "dirichlet") and (bc_types["top"] == "dirichlet"):
            bc = 3
        elif (bc_types["left"] == "dirichlet") and (bc_types["right"] == "dirichlet") and (bc_types["bottom"] == "neumann") and (bc_types["top"] == "dirichlet"):
            bc = 4
        elif (bc_types["left"] == "dirichlet") and (bc_types["right"] == "dirichlet") and (bc_types["bottom"] == "dirichlet") and (bc_types["top"] == "neumann"):
            bc = 5
        else:
            raise ValueError("This combination of boundary conditions is not implemented for the exact solution of the 2D Laplace equation.")

        # Initialize solutions
        if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
            u_L = 0
            u_R = 0
            u_T = 0
            u_B = 0

        # the solution method yields a sum to N = infinity
        # we need to sum over a finite number
        # This may need to be increased to reduce oscillations
        N = 5

        for n in range(1, N+1):
            lambda_n = n*np.pi/H
            mu_n = n*np.pi/L

            if bc_func_L == "quadratic":
                # quadratic boundary term
                # gL = aL + bL*(y-cL) + dL*(y-eL)^2
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^H gL(y)*sin(lambda_n*y) dy
                    boundary_integral = (-2*dL + (aL - bL*cL + dL*eL**2)*lambda_n**2 - (dL*(-2 + lambda_n**2*(eL - H)**2) + lambda_n**2*(aL + bL*(-cL + H)))*sp.cos(lambda_n*H) + lambda_n*(bL + 2*dL*(-eL + H))*sp.sin(lambda_n*H))/lambda_n**3
            elif bc_func_L == "sine":
                # sine boundary term
                # gL = aL + bL*sin(cL*y)
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^H gL(y)*sin(lambda_n*y) dy
                    if np.abs(cL-lambda_n) > abs(cL)/10000:
                        boundary_integral = (aL - aL*sp.cos(lambda_n*H))/lambda_n + (bL*(lambda_n*sp.cos(lambda_n*H)*sp.sin(cL*H) - cL*sp.cos(cL*H)*sp.sin(lambda_n*H)))/(cL**2 - lambda_n**2)
                    else:  # for special case cL = lambda_n we need a different solution
                        boundary_integral = (4*aL + 2*bL*lambda_n*H - 4*aL*sp.cos(lambda_n*H) - bL*sp.sin(2*lambda_n*H))/(4*lambda_n)
            elif bc_func_L == "cosine":
                # cosine boundary term
                # gL = aL + bL*cos(bL*y)
                # \int_0^H gL(y)*sin(lambda_n*y) dy
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    if np.abs(cL-lambda_n) > abs(cL)/10000:
                        boundary_integral = (aL - aL*sp.cos(lambda_n*H))/lambda_n + (bL*(-lambda_n + lambda_n*sp.cos(cL*H)*sp.cos(lambda_n*H) + cL*sp.sin(cL*H)*sp.sin(lambda_n*H)))/(cL**2 - lambda_n**2)
                    else:
                        boundary_integral = ((2*aL + bL + bL*sp.cos(lambda_n*H))*sp.sin((lambda_n*H)/2)**2)/lambda_n
            if bc == 1 or bc == 3 or bc == 4 or bc == 5:
                c_L_n = (2/H)*(1/sp.sinh(lambda_n*(-L)))*boundary_integral
                u_L = u_L + c_L_n*sp.sinh(lambda_n*(x-L))*sp.sin(lambda_n*y)
            elif bc == 2:
                c_L_n = -(2/H)*(1/(lambda_n*sp.cosh(lambda_n*(-L))))*boundary_integral  # minus sign due to boundary condition being set for gradient normal to wall
                u_L = u_L + c_L_n*sp.sinh(lambda_n*(x-L))*sp.sin(lambda_n*y)

            if bc_func_R == "quadratic":
                # quadratic boundary term
                # gR = aR + bR*(y-cR) + dR*(y-eR)^2
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^H gR(y)*sin(lambda_n*y) dy
                    boundary_integral = (-2*dR + (aR - bR*cR + dR*eR**2)*lambda_n**2 - (dR*(-2 + lambda_n**2*(eR - H)**2) + lambda_n**2*(aR + bR*(-cR + H)))*sp.cos(lambda_n*H) + lambda_n*(bR + 2*dR*(-eR + H))*sp.sin(lambda_n*H))/lambda_n**3
            elif bc_func_R == "sine":
                # sine boundary term
                # gR = aR + bR*sin(cR*y)
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^H gR(y)*sin(lambda_n*y) dy
                    if np.abs(cR-lambda_n) > abs(cR)/10000:
                        boundary_integral = (aR - aR*sp.cos(lambda_n*H))/lambda_n + (bR*(lambda_n*sp.cos(lambda_n*H)*sp.sin(cR*H) - cR*sp.cos(cR*H)*sp.sin(lambda_n*H)))/(cR**2 - lambda_n**2)
                    else:  # for special case cR = lambda_n we need a different solution
                        boundary_integral = (4*aR + 2*bR*lambda_n*H - 4*aR*sp.cos(lambda_n*H) - bR*sp.sin(2*lambda_n*H))/(4*lambda_n)
            elif bc_func_R == "cosine":
                # cosine boundary term
                # gR = aR + bR*cos(cR*y)
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^H gR(y)*sin(lambda_n*y) dy
                    if np.abs(cR-lambda_n) > abs(cR)/10000:
                        boundary_integral = (aR - aR*sp.cos(lambda_n*H))/lambda_n + (bR*(-lambda_n + lambda_n*sp.cos(cR*H)*sp.cos(lambda_n*H) + cR*sp.sin(cR*H)*sp.sin(lambda_n*H)))/(cR**2 - lambda_n**2)
                    else:
                        boundary_integral = ((2*aR + bR + bR*sp.cos(lambda_n*H))*sp.sin((lambda_n*H)/2)**2)/lambda_n
            if bc == 1 or bc == 2 or bc == 4 or bc == 5:
                c_R_n = (2/H)*(1/sp.sinh(lambda_n*L))*boundary_integral
                u_R = u_R + c_R_n*sp.sinh(lambda_n*x)*sp.sin(lambda_n*y)
            elif bc == 3:
                c_R_n = (2/H)*(1/(lambda_n*sp.cosh(lambda_n*L)))*boundary_integral
                u_R = u_R + c_R_n*sp.sinh(lambda_n*x)*sp.sin(lambda_n*y)

            if bc_func_B == "quadratic":
                # quadratic boundary term
                # gB = aB + bB*(x-cB) + dB*(x-eB)^2
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^L gB(x)*sin(mu_n*x) dx
                    boundary_integral = boundary_integral = (-2*dB + (aB - bB*cB + dB*eB**2)*mu_n**2 - (dB*(-2 + mu_n**2*(eB - L)**2) + mu_n**2*(aB + bB*(-cB + L)))*sp.cos(mu_n*L) + mu_n*(bB + 2*dB*(-eB + L))*sp.sin(mu_n*L))/mu_n**3
            elif bc_func_B == "sine":
                # sine boundary term
                # gB = aB + bB*sin(cB*x)
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^L gB(x)*sin(mu_n*x) dx
                    if np.abs(cB-mu_n) > abs(cB)/10000:
                        boundary_integral = (aB - aB*sp.cos(mu_n*L))/mu_n + (bB*(mu_n*sp.cos(mu_n*L)*sp.sin(cB*L) - cB*sp.cos(cB*L)*sp.sin(mu_n*L)))/(cB**2 - mu_n**2)
                    else:  # for special case cB = mu_n we need a different solution
                        boundary_integral = (4*aB + 2*bB*mu_n*L - 4*aB*sp.cos(mu_n*L) - bB*sp.sin(2*mu_n*L))/(4*mu_n)
            elif bc_func_B == "cosine":
                # cosine boundary term
                # gB = aB + bB*cos(cB*y)
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^L gB(x)*sin(mu_n*x) dx
                    if np.abs(cB-mu_n) > abs(cB)/10000:
                        boundary_integral = (aB - aB*sp.cos(mu_n*L))/mu_n + (bB*(-mu_n + mu_n*sp.cos(cB*L)*sp.cos(mu_n*L) + cB*sp.sin(cB*L)*sp.sin(mu_n*L)))/(cB**2 - mu_n**2)
                    else:
                        boundary_integral = ((2*aB + bB + bB*sp.cos(mu_n*L))*sp.sin((mu_n*L)/2)**2)/mu_n
            if bc == 1 or bc == 2 or bc == 3 or bc == 5:
                c_B_n = (2/L)*(1/sp.sinh(mu_n*(-H)))*boundary_integral
                u_B = u_B + c_B_n*sp.sinh(mu_n*(y-H))*sp.sin(mu_n*x)
            elif bc == 4:
                c_B_n = -(2/L)*(1/(mu_n*sp.cosh(mu_n*(-H))))*boundary_integral
                u_B = u_B + c_B_n*sp.sinh(mu_n*(y-H))*sp.sin(mu_n*x)

            if bc_func_T == "quadratic":
                # quadratic boundary term
                # gT = aT + bT*(x-cT) + dT*(x-eT)^2
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^L gT(x)*sin(mu_n*x) dx
                    boundary_integral = (-2*dT + (aT - bT*cT + dT*eT**2)*mu_n**2 - (dT*(-2 + mu_n**2*(eT - L)**2) + mu_n**2*(aT + bT*(-cT + L)))*sp.cos(mu_n*L) + mu_n*(bT + 2*dT*(-eT + L))*sp.sin(mu_n*L))/mu_n**3
            elif bc_func_T == "sine":
                # sine boundary term
                # gT = aT + bT*sin(cT*x)
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^L gT(x)*sin(mu_n*x) dx
                    if np.abs(cT-mu_n) > abs(cT)/10000:
                        boundary_integral = (aT - aT*sp.cos(mu_n*L))/mu_n + (bT*(mu_n*sp.cos(mu_n*L)*sp.sin(cT*L) - cT*sp.cos(cT*L)*sp.sin(mu_n*L)))/(cT**2 - mu_n**2)
                    else:  # for special case cT = mu_n we need a different solution
                        boundary_integral = (4*aT + 2*bT*mu_n*L - 4*aT*sp.cos(mu_n*L) - bT*sp.sin(2*mu_n*L))/(4*mu_n)
            elif bc_func_T == "cosine":
                # cosine boundary term
                # gT = aT + bT*cos(cT*y)
                if bc == 1 or bc == 2 or bc == 3 or bc == 4 or bc == 5:
                    # \int_0^L gT(x)*sin(mu_n*x) dx
                    if np.abs(cT-mu_n) > abs(cT)/10000:
                        boundary_integral = (aT - aT*sp.cos(mu_n*L))/mu_n + (bT*(-mu_n + mu_n*sp.cos(cT*L)*sp.cos(mu_n*L) + cT*sp.sin(cT*L)*sp.sin(mu_n*L)))/(cT**2 - mu_n**2)
                    else:
                        boundary_integral = ((2*aT + bT + bT*sp.cos(mu_n*L))*sp.sin((mu_n*L)/2)**2)/mu_n
            if bc == 1 or bc == 2 or bc == 3 or bc == 4:
                c_T_n = (2/L)*(1/sp.sinh(mu_n*H))*boundary_integral
                u_T = u_T + c_T_n*sp.sinh(mu_n*y)*sp.sin(mu_n*x)
            elif bc == 5:
                c_T_n = (2/L)*(1/(mu_n*sp.cosh(mu_n*H)))*boundary_integral
                u_T = u_T + c_T_n*sp.sinh(mu_n*y)*sp.sin(mu_n*x)

        # combine the four solution components, which are associated with the four boundaries
        u = u_L + u_R + u_T + u_B

        return u, x, y
