# Quadratic program

import numpy as np
import cvxpy as cp

# Genetate a random non-trivial qaudratic program
m = 15
n = 10
p = 5
np.random.seed(1)

P = np.random.randn(n, n)
P = P.T @ P  # Semidefinite matrix
q = np.random.randn(n)
G = np.random.randn(m, n)
h = G @ np.random.randn(n)  # Be careful with matrix dimension
A = np.random.randn(p, n)
b = np.random.randn(p)

# Define nd solve the CVXPY problem
x = cp.Variable(n)
constraints = [G @ x <= h,
              A @ x == b]
# x.T @ P @ x == cp.quad_form(x, P)
prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x), constraints)
prob.solve()  # Default solver for QP is 'OSQP'

# Print result.
print("\nsolver:",prob.solver_stats.solver_name)
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)

"""
Result
solver: OSQP
The optimal value is 86.891415855699
A solution x is
[-1.68244521  0.29769913 -2.38772183 -2.79986015  1.18270433 -0.20911897
 -4.50993526  3.76683701 -0.45770675 -3.78589638]
A dual solution corresponding to the inequality constraints is
[ 0.          0.          0.          0.          0.         10.45538054
  0.          0.          0.         39.67365045  0.          0.
  0.         20.79927156  6.54115873]
"""
