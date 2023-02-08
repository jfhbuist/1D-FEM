[![Build Status](https://github.com/jfhbuist/flexible-fem/actions/workflows/CI.yml/badge.svg?event=push)](https://github.com/jfhbuist/flexible-fem/actions)
[![codecov](https://codecov.io/gh/jfhbuist/flexible-fem/branch/main/graph/badge.svg?token=BFUOZDUQ6G)](https://codecov.io/gh/jfhbuist/flexible-fem)

# flexible-fem

This is a one- or two-dimensional standard Galerkin finite element code.
It can solve steady state scalar equations, including advection, diffusion, and reaction operators, along with source terms.
Implemented boundary conditions include inhomogeneous Dirichlet and inhomogeneous Neumann boundary conditions. 

To use this package, first install it (in a virtual environment):
```
pip install .
```

Next, run one of the examples:
```
python examples/laplace_2D.py
```

These examples can be modified to use different parameters and boundary conditions.
They run problems defined in "fem_front.py", which uses the functionality given in "fem_core.py".
Numerical solutions can be compared to the exact solutions given in "exact.py". 
