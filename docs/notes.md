# Notes

## Done

- Improve structure with classes and wrapping of functions
- Compute exact solution and compare to numerical solution
- Make basis functions functions of x, and do transformation to xi in generate_element_matrix
- Wrap FEM workflow in function 
- Allow computation of u at arbitrary x (using basis functions):  make FEM solution an evaluable function of x (like exact solution)
- Generalize to different boundary conditions (inhomogeneous Neumann, inhomogeneous Dirichlet)
- Add stiffness matrix for linear advection operator
- Generalize to different source functions -> Define source in fem_front
- Give code a nice package structure, with CI (testing)
- Rename basis functions to shape functions (N1, N2, N3)
- Change structure of code to use master elements and jacobian (isoparametric mapping)
- Fix bc/bc_params inconsistency
- Make integrand calculation dimension-independent
- Define bc functions in front
- Fix 2D integration so it is over triangle instead of square
- Solution reconstruction was broken. The check if a point was in a certain element was incorrect: it actually checked if a point was in a square twice the size of the triangle. Fix this.
- Make FEM multidimensional (2D) - create grid, check boundary conditions, implement Laplace_2D test case
- Cleanup commented and unnecessary code
- Expand Laplace 2D exact calculation to Neumann boundary conditions
- Fix Neumann BC for 2D: fix coordinate transformation for boundary elements (required for natural boundary)

## Todo
   
- Rewrite natural boundary implementation to use coordinate transformation and intgration of the element corresponding to the boundary element?
- Separate grid generation: save a grid after it is generated. Add a matrix (vector) linking boundary elements to proper elements.
- Add more exact solutions for 2D Laplace with Neumann BC
- Refactoring: check that natural boundary terms, source terms, and stiffness matrix construction follow the same pattern
- Add more tests for 2D
- Use problems from SCPDE as test cases
- Check which part of code is slow and make it more efficient. Replace python arrays by numpy arrays and vectorize code?
- Merge source and natural boundary operators?
- Add test cases from literature
- Add mass matrix for time derivative
- Add exact solution to (time-dependent) heat equation
- Generalize to non-uniform grids 
- Add stiffness matrix for nonlinear advection operator
- Implement varying D, R, A, and put D inside derivative
- Coordinate_transformation_inverse can be replaced by just defining dimensional basis functions (could make code more efficient?)
- Define vertex basis functions (in addition to elements)? Could be useful for solution reconstruction.
- Change method of solution reconstruction. Construct global solution as a function, by taking the sum of each basis function, multiplied with its coefficient. 
- Give code better header and name
- Improve docstrings and add automatic documentation 
- Implement systematic calculation of stiffness matrix and source vector using constraints on basis functions

## Literature

- Riley & Hobson (2011), Essential Mathematical Methods for the Physical Sciences, chapter 6 
- Paul's Online Notes: https://tutorial.math.lamar.edu/Classes/DE/LaplacesEqn.aspx
 
- Hirsch (2007), Numerical Computation of Internal & External Flows, chapter A.5.4
- Peiro and Sherwin, Finite Difference, Finite Element, and Finite Volume Methods for Partial Differential Equations, 2005
- Segal, Finite element methods for the incompressible Navier-Stokes equations, 2021
- Langtangen & Logg, Solving PDEs in Python: The FEniCS Tutorial I, 2016
  
- Cuneyt Sert ME 582 Finite Element Analysis in Thermofluids: https://users.metu.edu.tr/home204/csert/wwwhome/teaching_notes.htm
- Wolfgang Bangerth: https://www.math.colostate.edu/~bangerth/videos.html (lectures 21.6 and 21.65) 
- Hans Langtangen, Introduction to finite element methods: http://hplgit.github.io/INF5620/doc/pub/H14/fem/html/main_fem.html
- what-when-how: http://what-when-how.com/the-finite-element-method/fem-for-two-dimensional-solids-finite-element-method-part-1/
