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
    def get_solution(self, pde, bc, L, n, params):
        if pde == 'steady_diffusion_reaction_1D':
            u_sym, x_sym = self.steady_diffusion_reaction_1D(bc, L, params["D"], params["R"], params["alpha"], params["beta"], params["gamma"])
        if pde == 'steady_advection_diffusion_reaction_1D':
            u_sym, x_sym = self.steady_advection_diffusion_reaction_1D(bc, L, params["A"], params["D"], params["R"], params["alpha"], params["beta"], params["gamma"])
        if pde == 'steady_advection_diffusion_1D':
            u_sym, x_sym = self.steady_advection_diffusion_1D(bc, L, params["A"], params["D"], params["alpha"], params["beta"], params["gamma"])
        # convert to a python function:
        u_num = broadcast(sp.lambdify(x_sym, u_sym, "numpy")) 
        # specify points at which to return function:
        x = np.linspace(0,L,n) 
        # evalute function at specified points: 
        u = u_num(x) 
        return u, x
            
    def steady_diffusion_reaction_1D(self, bc, L, D, R, alpha, beta, gamma):
        # Diffusion-reaction equation (aka Helmholtz equation):
        # -D*u_xx + R*u = f
        # with source term:
        # f = alpha + beta*sin(gamma*x)
        
        # see chapter 6, Riley & Hobson (2011)
        # x is our symbolic spatial variable:
        x = sp.symbols("x")
        # periodic source terms:
        f = alpha + beta*sp.sin(gamma*x)
        
        # complementary solution (general solution to the homogeneous equation):
        # uc = c0*exp(mu_0*x) + c1*exp(mu_1*x)  
        mu_0 = np.sqrt(R/D) 
        mu_1 = -np.sqrt(R/D) 
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
    
    def steady_advection_diffusion_reaction_1D(self, bc, L, A, D, R, alpha, beta, gamma):
        # Advection-diffusion-reaction equation
        # A*u_x - D*u_xx + R*u = f
        # with source term:
        # f = alpha + beta*sin(gamma*x)
        
        # see chapter 6, Riley & Hobson (2011)
        # x is our symbolic spatial variable:
        x = sp.symbols("x")
        # periodic source terms:
        f = alpha + beta*sp.sin(gamma*x)
        
        # complementary solution (general solution to the homogeneous equation):
        # uc = c0*exp(mu_0*x) + c1*exp(mu_1*x)  
        mu_0 = (-A + np.sqrt(A**2 + 4*D*R))/(-2*D) 
        mu_1 = (-A - np.sqrt(A**2 + 4*D*R))/(-2*D) 
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
    
    def steady_advection_diffusion_1D(self, bc, L, A, D, alpha, beta, gamma):
        # Advection-diffusion equation
        # A*u_x - D*u_xx = f
        # with source term:
        # f = alpha + beta*sin(gamma*x)
        
        # see chapter 6, Riley & Hobson (2011)
        # x is our symbolic spatial variable:
        x = sp.symbols("x")
        # periodic source terms:
        f = alpha + beta*sp.sin(gamma*x)
        
        # complementary solution (general solution to the homogeneous equation):
        # uc = c0*exp(mu_0*x) + c1*exp(mu_1*x)  
        mu_0 = 0
        mu_1 = A/D
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
            # L = 1
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
        