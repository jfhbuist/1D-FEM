# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:23:49 2022

@author: jurri
"""

import numpy as np
import sympy as sp

def broadcast(fun):
    # this is needed when using lambdify, 
    # to make a constant function return a vector when the input is a vector
    return lambda *x: np.broadcast_arrays(fun(*x), *x)[0]

class ExactSolution:
    def get_solution(self, pde, bc, bc_params, grid_params, core_params, source_params):
        if pde == 'steady_diffusion_reaction_1D':
            u_sym, x_sym = self.steady_diffusion_reaction_1D(bc, bc_params, grid_params, core_params, source_params)
            dim = 1
        elif pde == 'steady_advection_diffusion_reaction_1D':
            u_sym, x_sym = self.steady_advection_diffusion_reaction_1D(bc, bc_params, grid_params, core_params, source_params)
            dim = 1
        elif pde == 'steady_advection_diffusion_1D':
            u_sym, x_sym = self.steady_advection_diffusion_1D(bc, bc_params, grid_params, core_params, source_params)
            dim = 1
        elif pde == 'laplace_1D':
            u_sym, x_sym = self.laplace_1D(bc, bc_params, grid_params, core_params, source_params)
            dim = 1
        elif pde == 'laplace_2D':
            u_sym, x_sym, y_sym = self.laplace_2D(bc, bc_params, grid_params, core_params, source_params)
            dim = 2
        if dim == 1:
            L = grid_params["L"]
            n = grid_params["n"]
            # convert to a python function:
            u_num = broadcast(sp.lambdify(x_sym, u_sym, "numpy")) 
            # specify points at which to return function:
            x = np.linspace(0,L,n) 
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
            x = np.linspace(0,L,nx) 
            y = np.linspace(0,H,ny) 
            X, Y = np.meshgrid(x, y)
            # evaluate function at specified points: 
            U = u_num(X, Y) 
            return U, X, Y

            
    def steady_diffusion_reaction_1D(self, bc, bc_params, grid_params, core_params, source_params):
        # Diffusion-reaction equation (aka Helmholtz equation):
        # -D*u_xx + R*u = f
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
            # particular solution (this may be any solution to the full inhomogeneous equation, and is source dependent):
            up = alpha/R + beta*sp.sin(gamma*x)/(R + D*gamma**2)
            
        dupdx = sp.diff(up,x) # needed for neumann boundary conditions
        # general solution is the sum of the two:
        # u = uc + up
        # constants are set afterwards through boundary conditions
        
        c0, c1 = self.solve_for_bc_1D(bc, L, mu_0, mu_1, x, up, dupdx)        
        uc = c0*sp.exp(mu_0*x) + c1*sp.exp(mu_1*x)  
        u = uc + up # this is a symbolic function
        return u, x
    
    def steady_advection_diffusion_reaction_1D(self, bc, bc_params, grid_params, core_params, source_params):
        # Advection-diffusion-reaction equation
        # A*u_x - D*u_xx + R*u = f
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
            # particular solution (this may be any solution to the full inhomogeneous equation, and is source dependent):
            b0 = alpha/R
            b1 = beta/(R + D*gamma**2) -beta*((A*gamma)/(R + D*gamma**2))*((A*gamma)/((A*gamma)**2 + (R + D*gamma**2)**2))
            b2 = -beta*(A*gamma)/(((A*gamma)**2) + (R + D*gamma**2)**2)
            up = b0 + b1*sp.sin(gamma*x) + b2*sp.cos(gamma*x)
        
        dupdx = sp.diff(up,x) # needed for neumann boundary conditions
        # general solution is the sum of the two:
        # u = uc + up
        # constants are set afterwards through boundary conditions
        
        c0, c1 = self.solve_for_bc_1D(bc, L, mu_0, mu_1, x, up, dupdx)        
        uc = c0*sp.exp(mu_0*x) + c1*sp.exp(mu_1*x)  
        u = uc + up # this is a symbolic function
        return u, x
    
    def steady_advection_diffusion_1D(self, bc, bc_params, grid_params, core_params, source_params):
        # Advection-diffusion equation
        # A*u_x - D*u_xx = f
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
            # particular solution (this may be any solution to the full inhomogeneous equation, and is source dependent):
            b0 = 0
            b1 = beta/(D*gamma**2) - beta*((A*gamma)/(D*gamma**2))*((A*gamma)/((A*gamma)**2 + (D*gamma**2)**2))
            b2 = -beta*(A*gamma)/((A*gamma)**2 + (D*gamma**2)**2)
            b3 = alpha/A
            up = b0 + b1*sp.sin(gamma*x) + b2*sp.cos(gamma*x) + b3*x
        
        dupdx = sp.diff(up,x) # needed for neumann boundary conditions
        # general solution is the sum of the two:
        # u = uc + up
        # constants are set afterwards through boundary conditions
        
        c0, c1 = self.solve_for_bc_1D(bc, L, mu_0, mu_1, x, up, dupdx)        
        uc = c0*sp.exp(mu_0*x) + c1*sp.exp(mu_1*x)  
        u = uc + up # this is a symbolic function
        return u, x 
    
    def solve_for_bc_1D(self, bc, L, mu_0, mu_1, x, up, dupdx):
        # we need to solve for c0 and c1 using the boundary conditions
        # we will get an equation of the form A*c = B, with c = [c0; c1].
        A = np.zeros((2,2))
        B = np.zeros(2)
        if bc["left"][0] == "neumann":
            # mu_0*c0 + mu_1*c1 + dupdx(0) = bc(left)
            A[0,0] = mu_0
            A[0,1] = mu_1
            B[0] = -dupdx.subs(x,0) + bc["left"][1]
        elif bc["left"][0] == "dirichlet":
            # c0 + c1 + up(0) = bc(left)
            A[0,0] = 1
            A[0,1] = 1
            B[0] = -up.subs(x,0) + bc["left"][1]         
        if bc["right"][0] == "neumann":
            # mu_0*c0*exp(mu_0*L) + mu_1*c1*exp(mu_1*L) + dupdx(1) = bc(right)
            A[1,0] = mu_0*np.exp(mu_0*L)
            A[1,1] = mu_1*np.exp(mu_1*L) 
            B[1] = -dupdx.subs(x,L) + bc["right"][1]
        elif bc["right"][0] == "dirichlet":
            # c0*exp(mu_0*L) + c1*exp(mu_1*L) + up(L) = bc(right)
            A[1,0] = np.exp(mu_0*L)
            A[1,1] = np.exp(mu_1*L)
            B[1] = -up.subs(x,L) + bc["right"][1]  
        c = np.linalg.solve(A, B)
        c0 = c[0]
        c1 = c[1]
        return c0, c1
    
    def laplace_1D(self, bc, bc_params, grid_params, core_params, source_params):
        # Laplace equation:
        # -D*u_xx = 0
        D = core_params["D"]
        L = grid_params["L"]
        
        # see chapter 6, Riley & Hobson (2011)
        # x is our symbolic spatial variable:
        x = sp.symbols("x")
        
        # complementary solution (general solution to the homogeneous equation):
        # uc = c0 + c1*x
              
        # no source terms, so no particular solution
        # u = uc 
        # constants are set through boundary conditions
        
        A = np.zeros((2,2))
        B = np.zeros(2)
        if bc["left"][0] == "neumann":
            # c1 = bc(left)
            A[0,0] = 0
            A[0,1] = 1
            B[0] = bc["left"][1]
        elif bc["left"][0] == "dirichlet":
            # c0 = bc(left)
            A[0,0] = 1
            A[0,1] = 0
            B[0] = bc["left"][1]         
        if bc["right"][0] == "neumann":
            # c1 = bc(right)
            A[1,0] = 0
            A[1,1] = 1
            B[1] = bc["right"][1]
        elif bc["right"][0] == "dirichlet":
            # c0 + c1*L = bc(right)
            A[1,0] = 1
            A[1,1] = L
            B[1] = bc["right"][1]  
        c = np.linalg.solve(A, B)
        c0 = c[0]
        c1 = c[1]
        
        uc = c0 + c1*x 
        u = uc 
        return u, x
    

    
    def laplace_2D(self, bc, bc_params, grid_params, core_params, source_params):
        # Laplace equation:
        # -D*(u_xx + u_yy) = 0
        D = core_params["D"]
        L = grid_params["L"]
        H = grid_params["H"]
        
        bc_func_0 = bc_params["left"][0]
        a0 = bc_params["left"][1]
        b0 = bc_params["left"][2]
        c0 = bc_params["left"][3]
        d0 = bc_params["left"][4]
        e0 = bc_params["left"][5]
        
        bc_func_1 = bc_params["right"][0]
        a1 = bc_params["right"][1]
        b1 = bc_params["right"][2]
        c1 = bc_params["right"][3]
        d1 = bc_params["right"][4]
        e1 = bc_params["right"][5]
        
        bc_func_2 = bc_params["upper"][0]
        a2 = bc_params["upper"][1]
        b2 = bc_params["upper"][2]
        c2 = bc_params["upper"][3]
        d2 = bc_params["upper"][4]
        e2 = bc_params["upper"][5]
        
        bc_func_3 = bc_params["lower"][0]
        a3 = bc_params["lower"][1]
        b3 = bc_params["lower"][2]
        c3 = bc_params["lower"][3]
        d3 = bc_params["lower"][4]
        e3 = bc_params["lower"][5]
        
        # see chapter 11, Riley & Hobson (2011)
        # and https://tutorial.math.lamar.edu/Classes/DE/LaplacesEqn.aspx
        # x and y are our symbolic spatial variables:
        x = sp.symbols("x")
        y = sp.symbols("y")
        
        # separation of variables:
        # X'' = lambda^2*X, Y'' = -lambda^2*Y
        # general separated solution:
        # u = X(x)*Y(y)
        # X(x) = (A*cosh(lambda*x) + B*sinh(lambda*x))
        # Y(y) = C*cos(lambda*y)+D*sin(lambda*y)) 
               
        # we solve the boundary value problem in 4 parts
        
        # for the left boundary, we take
        # X(x) = A*cosh(lambda*(x-L)) + B*sinh(lambda*(x-L))
        # Y(y) = C*cos(lambda*y) + D*sin(lambda*y) 
        # lambda_n = n*pi/H
        # Assume homogeneous dirichlet bc everywhere, except at left boundary u = g0(y)
        # A = 0, C = 0
        # u_0 = \sum_n c_0_n*sinh(lambda_n*(x-L))*sin(lambda_n*y)
        # c_0_n = (2/H)*(1/sinh(lambda_n*(-L)))*\int_0^H g0(y)*sin(lambda_n*y) dy
              
        # for the right boundary, we take
        # X(x) = A*cosh(lambda*x) + B*sinh(lambda*x)
        # Y(y) = C*cos(lambda*y) + D*sin(lambda*y) 
        # lambda_n = n*pi/H
        # Assume homogeneous dirichlet bc everywhere, except at left boundary u = g1(y)
        # A = 0, C = 0
        # u_1 = \sum_n c_1_n*sinh(lambda_n*x)*sin(lambda_n*y)
        # c_1_n = (2/H)*(1/sinh(lambda_n*L))*\int_0^H g1(y)*sin(lambda_n*y) dy
        
        # for the upper boundary, we take
        # X(x) = A*cos(mu*x) + B*sin(mu*x) 
        # Y(y) = C*cosh(mu*y) + D*sinh(mu*y)
        # mu_n = n*pi/L
        # Assume homogeneous dirichlet bc everywhere, except at upper boundary u = g2(x)
        # A = 0, C = 0
        # u_2 = \sum_n c_2_n*sinh(mu_n*x)*sin(mu_n*x)
        # c_2_n = (2/L)*(1/sinh(mu_n*H)*\int_0^L g2(x)*sin(mu_n*x) dx
        
        # for the lower boundary, we take
        # X(x) = A*cos(mu*x) + B*sin(mu*x) 
        # Y(y) = C*cosh(mu*(y-H)) + D*sinh(mu*(y-H))
        # mu_n = n*pi/L
        # Assume homogeneous dirichlet bc everywhere, except at lower boundary u = g3(x0)
        # A = 0, C = 0
        # u_3 = \sum_n c_3_n*sinh(mu_n*(y-H))*sin(mu_n*x)
        # c_3_n = (2/L)*(1/sinh(mu_n*(-H))*\int_0^L g3(x)*sin(mu_n*x) dx
        
                 
        N=5
        u_0 = 0     
        u_1 = 0
        u_2 = 0
        u_3 = 0
        # the solution method yields a sum to N = infinity
        # we need to sum over a finite number
        for n in range(1,N+1):
            lambda_n = n*np.pi/H
            mu_n = n*np.pi/L
            
            #if  bc_func_0 == "periodic":
                # periodic boundary term
                # g0 = a + b*sin(c*y) + d*cos(e*y)
                # \int_0^H g0(y)*sin(lambda_n*y) dy
                #boundary_integral = (a0 - a0*sp.cos(lambda_n*H))/lambda_n + (b0*(lambda_n*sp.cos(lambda_n*H)*sp.sin(c0*H) - c0*sp.cos(c0*H)*sp.sin(lambda_n*H)))/(c0**2 - lambda_n**2) + (d0*(-lambda_n + lambda_n*sp.cos(e0*H)*sp.cos(lambda_n*H) + e0*sp.sin(e0*H)*sp.sin(lambda_n*H)))/(e0**2 - lambda_n**2)
                #boundary_integral = (d0*e0*lambda_n*sp.sin(e0*x1)*sp.sin(lambda_n*x1) + sp.cos(lambda_n*x1)*(d0*lambda_n**2*sp.cos(lambda_n*x1) - (e0**2 - lambda_n**2)*(a0 + b0*sp.sin(c0*x1))))/((e0 - lambda_n)*lambda_n*(e0 + lambda_n))
                #-(a0*sp.cos(lambda_n*x))/lambda_n + (b0*sp.sin((c0-lambda_n)*x1))/(2*(c0-lambda_n)) - (b0*sp.sin((c0+lambda_n)*x1))/(2*(c0+lambda_n))
            if bc_func_0 == "quadratic":
                # quadratic boundary term
                # g0 = a0 + b0*(y-c0) + d0*(y-e0)^2 
                # \int_0^H g0(y)*sin(lambda_n*y) dy
                boundary_integral = (-2*d0 + (a0 - b0*c0 + d0*e0**2)*lambda_n**2 - (d0*(-2 + lambda_n**2*(e0 - H)**2) + lambda_n**2*(a0 + b0*(-c0 + H)))*sp.cos(lambda_n*H) + lambda_n*(b0 + 2*d0*(-e0 + H))*sp.sin(lambda_n*H))/lambda_n**3
                # boundary_integral = (-((lambda_n**2*(a0 - b0*c0 + b0*x1) + d0*(-2 + e0**2*lambda_n**2 - 2*e0*lambda_n**2*x1 + lambda_n**2*x1**2))*sp.cos(lambda_n*x1)) + lambda_n*(b0 + 2*d0*(-e0 + x1))*sp.sin(lambda_n*x1))/lambda_n**3
                #boundary_integral = (-lambda_n*(a0+b0*(-c0+x1))*sp.cos(lambda_n*x1)+b0*sp.sin(d0*x1))/lambda_n**2
            elif bc_func_0 == "sine":
                # sine boundary term
                # g0 = a0 + b0*sin(c0*y) 
                # \int_0^H g0(y)*sin(lambda_n*y) dy
                if np.abs(c0-lambda_n) > abs(c0)/10000:
                    boundary_integral = (a0 - a0*sp.cos(lambda_n*H))/lambda_n + (b0*(lambda_n*sp.cos(lambda_n*H)*sp.sin(c0*H) - c0*sp.cos(c0*H)*sp.sin(lambda_n*H)))/(c0**2 - lambda_n**2)
                    #boundary_integral = (a0*(lambda_n*sp.cos(lambda_n*H)*sp.sin(b0*H) - b0*sp.cos(b0*H)*sp.sin(lambda_n*H)))/(b0**2 - lambda_n**2)
                else: # for special case c0 = lambda_n we need a different solution
                    boundary_integral = (4*a0 + 2*b0*lambda_n*H - 4*a0*sp.cos(lambda_n*H) - b0*sp.sin(2*lambda_n*H))/(4*lambda_n)
                    #boundary_integral = (-(a0*(-2*lambda_n*H + sp.sin(2*lambda_n*H))))/(4*lambda_n)     
            elif bc_func_0 == "cosine":
                # cosine boundary term
                # g0 = a0 + b0*cos(b0*y) 
                # \int_0^H g0(y)*sin(lambda_n*y) dy
                if np.abs(c0-lambda_n) > abs(c0)/10000:
                    boundary_integral = (a0 - a0*sp.cos(lambda_n*H))/lambda_n + (b0*(-lambda_n + lambda_n*sp.cos(c0*H)*sp.cos(lambda_n*H) + c0*sp.sin(c0*H)*sp.sin(lambda_n*H)))/(c0**2 - lambda_n**2)
                    # boundary_integral = (a0*(-lambda_n + lambda_n*sp.cos(b0*H)*sp.cos(lambda_n*H) + b0*sp.sin(b0*H)*sp.sin(lambda_n*H)))/(b0**2 - lambda_n**2)
                else: 
                    boundary_integral = ((2*a0 + b0 + b0*sp.cos(lambda_n*H))*sp.sin((lambda_n*H)/2)**2)/lambda_n
                    #boundary_integral = (a0*sp.sin(lambda_n*H)**2)/(2*lambda_n)
            if bc["left"][0] == "dirichlet":
                c_0_n = (2/H)*(1/sp.sinh(lambda_n*(-L)))*boundary_integral
                u_0 = u_0 + c_0_n*sp.sinh(lambda_n*(x-L))*sp.sin(lambda_n*y)
                
            if bc_func_1 == "quadratic":
                # quadratic boundary term
                # g1 = a1 + b1*(y-c1) + d1*(y-e1)^2 
                # \int_0^H g1(y)*sin(lambda_n*y) dy
               boundary_integral = (-2*d1 + (a1 - b1*c1 + d1*e1**2)*lambda_n**2 - (d1*(-2 + lambda_n**2*(e1 - H)**2) + lambda_n**2*(a1 + b1*(-c1 + H)))*sp.cos(lambda_n*H) + lambda_n*(b1 + 2*d1*(-e1 + H))*sp.sin(lambda_n*H))/lambda_n**3
            elif bc_func_1 == "sine":
                # sine boundary term
                # g1 = a1 + b1*sin(c1*y) 
                # \int_0^H g1(y)*sin(lambda_n*y) dy
                if np.abs(c1-lambda_n) > abs(c1)/10000:
                    boundary_integral = (a1 - a1*sp.cos(lambda_n*H))/lambda_n + (b1*(lambda_n*sp.cos(lambda_n*H)*sp.sin(c1*H) - c1*sp.cos(c1*H)*sp.sin(lambda_n*H)))/(c1**2 - lambda_n**2)
                    #boundary_integral = (a1*(lambda_n*sp.cos(lambda_n*H)*sp.sin(b1*H) - b1*sp.cos(b1*H)*sp.sin(lambda_n*H)))/(b1**2 - lambda_n**2)
                else: # for special case c1 = lambda_n we need a different solution
                    boundary_integral = (4*a1 + 2*b1*lambda_n*H - 4*a1*sp.cos(lambda_n*H) - b1*sp.sin(2*lambda_n*H))/(4*lambda_n)
                   # boundary_integral = (-(a1*(-2*lambda_n*H + sp.sin(2*lambda_n*H))))/(4*lambda_n)     
            elif bc_func_1 == "cosine":
                # cosine boundary term
                # g1 = a1 + b1*cos(c1*y) 
                # \int_0^H g1(y)*sin(lambda_n*y) dy
                if np.abs(c1-lambda_n) > abs(c1)/10000:
                    boundary_integral = (a1 - a1*sp.cos(lambda_n*H))/lambda_n + (b1*(-lambda_n + lambda_n*sp.cos(c1*H)*sp.cos(lambda_n*H) + c1*sp.sin(c1*H)*sp.sin(lambda_n*H)))/(c1**2 - lambda_n**2)
                    #boundary_integral = (a1*(-lambda_n + lambda_n*sp.cos(b1*H)*sp.cos(lambda_n*H) + b1*sp.sin(b1*H)*sp.sin(lambda_n*H)))/(b1**2 - lambda_n**2)
                else:
                    boundary_integral = ((2*a1 + b1 + b1*sp.cos(lambda_n*H))*sp.sin((lambda_n*H)/2)**2)/lambda_n
                    #boundary_integral = (a1*sp.sin(lambda_n*H)**2)/(2*lambda_n)
            if bc["right"][0] == "dirichlet":
                c_1_n = (2/H)*(1/sp.sinh(lambda_n*L))*boundary_integral
                u_1 = u_1 + c_1_n*sp.sinh(lambda_n*x)*sp.sin(lambda_n*y)
            
            if bc_func_2 == "quadratic":
                # quadratic boundary term
                # g2 = a2 + b2*(x-c2) + d2*(x-e2)^2 
                # \int_0^L g2(x)*sin(mu_n*x) dx
                boundary_integral = (-2*d2 + (a2 - b2*c2 + d2*e2**2)*mu_n**2 - (d2*(-2 + mu_n**2*(e2 - L)**2) + mu_n**2*(a2 + b2*(-c2 + L)))*sp.cos(mu_n*L) + mu_n*(b2 + 2*d2*(-e2 + L))*sp.sin(mu_n*L))/mu_n**3
            elif bc_func_2 == "sine":
                # sine boundary term
                # g2 = a2 + b2*sin(c2*x) 
                # \int_0^L g2(x)*sin(mu_n*x) dx
                if np.abs(c2-mu_n) > abs(c2)/10000:
                    boundary_integral = (a2 - a2*sp.cos(mu_n*L))/mu_n + (b2*(mu_n*sp.cos(mu_n*L)*sp.sin(c2*L) - c2*sp.cos(c2*L)*sp.sin(mu_n*L)))/(c2**2 - mu_n**2)
                    #boundary_integral = (a2*(mu_n*sp.cos(mu_n*L)*sp.sin(b2*L) - b2*sp.cos(b2*L)*sp.sin(mu_n*L)))/(b2**2 - mu_n**2)
                else: # for special case c2 = mu_n we need a different solution
                    boundary_integral = (4*a2 + 2*b2*mu_n*L - 4*a2*sp.cos(mu_n*L) - b2*sp.sin(2*mu_n*L))/(4*mu_n)
                    #boundary_integral = (-(a2*(-2*mu_n*L + sp.sin(2*mu_n*L))))/(4*mu_n)     
            elif bc_func_2 == "cosine":
                # cosine boundary term
                # g2 = a2 + b2*cos(c2*y) 
                # \int_0^L g2(x)*sin(mu_n*x) dx
                if np.abs(c2-mu_n) > abs(c2)/10000:
                    boundary_integral = (a2 - a2*sp.cos(mu_n*L))/mu_n + (b2*(-mu_n + mu_n*sp.cos(c2*L)*sp.cos(mu_n*L) + c2*sp.sin(c2*L)*sp.sin(mu_n*L)))/(c2**2 - mu_n**2)
                    #boundary_integral = (a2*(-mu_n + mu_n*sp.cos(b2*L)*sp.cos(mu_n*L) + b2*sp.sin(b2*L)*sp.sin(mu_n*L)))/(b2**2 - mu_n**2)
                else:
                    boundary_integral = ((2*a2 + b2 + b2*sp.cos(mu_n*L))*sp.sin((mu_n*L)/2)**2)/mu_n
                    #boundary_integral = (a2*sp.sin(mu_n*L)**2)/(2*mu_n)
            if bc["upper"][0] == "dirichlet":
                c_2_n = (2/L)*(1/sp.sinh(mu_n*H))*boundary_integral
                u_2 = u_2 + c_2_n*sp.sinh(mu_n*y)*sp.sin(mu_n*x)
              
            if bc_func_3 == "quadratic":
                # quadratic boundary term
                # g3 = a3 + b3*(x-c3) + d3*(x-e3)^2 
                # \int_0^L g3(x)*sin(mu_n*x) dx
                boundary_integral = boundary_integral = (-2*d3 + (a3 - b3*c3 + d3*e3**2)*mu_n**2 - (d3*(-2 + mu_n**2*(e3 - L)**2) + mu_n**2*(a3 + b3*(-c3 + L)))*sp.cos(mu_n*L) + mu_n*(b3 + 2*d3*(-e3 + L))*sp.sin(mu_n*L))/mu_n**3
            elif bc_func_3 == "sine":
                # sine boundary term
                # g3 = a3 + b3*sin(c3*x) 
                # \int_0^L g3(x)*sin(mu_n*x) dx
                if np.abs(c3-mu_n) > abs(c3)/10000:
                    boundary_integral = (a3 - a3*sp.cos(mu_n*L))/mu_n + (b3*(mu_n*sp.cos(mu_n*L)*sp.sin(c3*L) - c3*sp.cos(c3*L)*sp.sin(mu_n*L)))/(c3**2 - mu_n**2)
                    #boundary_integral = (a3*(mu_n*sp.cos(mu_n*L)*sp.sin(b3*L) - b3*sp.cos(b3*L)*sp.sin(mu_n*L)))/(b3**2 - mu_n**2)
                else: # for special case c3 = mu_n we need a different solution
                    boundary_integral = (4*a3 + 2*b3*mu_n*L - 4*a3*sp.cos(mu_n*L) - b3*sp.sin(2*mu_n*L))/(4*mu_n)
                    #boundary_integral = (-(a3*(-2*mu_n*L + sp.sin(2*mu_n*L))))/(4*mu_n)     
            elif bc_func_3 == "cosine":
                # cosine boundary term
                # g3 = a3 + b3*cos(c3*y) 
                # \int_0^L g3(x)*sin(mu_n*x) dx
                if np.abs(c3-mu_n) > abs(c3)/10000:
                    boundary_integral = (a3 - a3*sp.cos(mu_n*L))/mu_n + (b3*(-mu_n + mu_n*sp.cos(c3*L)*sp.cos(mu_n*L) + c3*sp.sin(c3*L)*sp.sin(mu_n*L)))/(c3**2 - mu_n**2)
                    #boundary_integral = (a3*(-mu_n + mu_n*sp.cos(b3*L)*sp.cos(mu_n*L) + b3*sp.sin(b3*L)*sp.sin(mu_n*L)))/(b3**2 - mu_n**2)
                else: 
                    boundary_integral = ((2*a3 + b3 + b3*sp.cos(mu_n*L))*sp.sin((mu_n*L)/2)**2)/mu_n
                    #boundary_integral = (a3*sp.sin(mu_n*L)**2)/(2*mu_n)
            if bc["lower"][0] == "dirichlet":
                c_3_n = (2/L)*(1/sp.sinh(mu_n*(-H)))*boundary_integral
                u_3 = u_3 + c_3_n*sp.sinh(mu_n*(y-H))*sp.sin(mu_n*x)
        
        # combine the four solution components, which are associated with the four boundaries
        u = u_0 + u_1 + u_2 + u_3
                  
        # dupdx = sp.diff(up,x) # needed for neumann boundary conditions
        # dupdy = sp.diff(up,y) # needed for neumann boundary conditions
        # general solution is the sum of the two:
        # u = uc + up
        # functions are determined afterwards through boundary conditions
        
  
        return u, x, y
        