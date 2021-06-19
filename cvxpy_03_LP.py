# Linear program

import cvxpy as cp
import numpy as np

# Generate a random non-trivial linear program
m = 15
n = 10
np.random.seed(1)
s0 = np.random.randn(m)  # normal distibution of random vector length 15
lamb0 = np.maximum(-s0, 0)  # dual variable, take negative values of s0 and make them positive value
s0 = np.maximum(s0, 0)  # take positive value and throw away negative values
x0 = np.random.randn(n)  # normal distibution of random vector length 10
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0  # .T: transpose

# Define and solve the CVXPY problem
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T@x),  # Linear objective functiob
                 [A @ x <= b])  # Affine inequality
# Solve and print result
prob.solve()  # OSQP
print("\nsolver:", prob.solver_stats.solver_name)
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)

prob.solve(solver=cp.CVXOPT)
print("\nsolver:", prob.solver_stats.solver_name)
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)

"""
Result
solver: ECOS
The optimal value is -15.220912605552897
A solution x is
[-1.10133381 -0.16360111 -0.89734939  0.03216603  0.6069123  -1.12687348
  1.12967856  0.88176638  0.49075229  0.8984822 ]
A dual solution is
[6.98804566e-10 6.11756416e-01 5.28171747e-01 1.07296862e+00
 3.93758849e-09 2.30153870e+00 4.25704004e-10 7.61206896e-01
 8.36905607e-09 2.49370377e-01 1.30187004e-09 2.06014070e+00
 3.22417207e-01 3.84054343e-01 1.59493641e-09]

solver: CVXOPT
The optimal value is -15.22091261568548
A solution x is
[-1.09814169 -0.18313413 -0.85421886  0.05440009  0.55358915 -1.06877671
  1.16297119  0.9256346   0.51673564  0.90373496]
A dual solution is
[9.16313434e-09 6.11756426e-01 5.28171755e-01 1.07296862e+00
 1.70895542e-08 2.30153872e+00 1.35012755e-09 7.61206896e-01
 3.00986151e-08 2.49370379e-01 4.69005842e-09 2.06014071e+00
 3.22417225e-01 3.84054320e-01 1.25662006e-08]
"""
