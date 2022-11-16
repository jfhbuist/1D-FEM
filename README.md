[![Build Status](https://github.com/jfhbuist/flexible-fem/actions/workflows/CI.yml/badge.svg?event=push)](https://github.com/jfhbuist/flexible-fem/actions)
[![codecov](https://codecov.io/gh/jfhbuist/flexible-fem/branch/main/graph/badge.svg?token=BFUOZDUQ6G)](https://codecov.io/gh/jfhbuist/flexible-fem)
# flexible-fem
This is a one-dimensional standard Galerkin finite element code.
It can solve steady state scalar equations, including advection, diffusion, and reaction operators, along with source terms.
Implemented boundary conditions include inhomogeneous Dirichlet and inhomogeneous Neumann boundary conditions. 

The program is run from the script "main.py". 
Here, the type of equation, the boundary conditions, and the input parameters are selected.
The flow of the program is dictated in "fem_front.py".
In this script, different combinations of the operators can be made to form different differential equations. 
The core of the finite element method is found in "fem_core.py".
FEM solutions can be compared to the exact solutions defined in "exact.py". 