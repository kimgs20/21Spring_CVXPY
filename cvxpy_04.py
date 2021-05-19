# Linear program

import cvxpy as cp
import numpy as np

# Generate a random non-trivial linear program
m = 15
n = 10
np.random.seed(1)
s0 = np.random.randn(m)  # what is s0?
lamb0 = np.maximum(-s0, 0)  # dual
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0  # .T: transpose

# Define and solve the CVXPY problem
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T@x),
                 [A @ x <= b])
prob.solve()

# Print result
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)
"""
Result
The optimal value is -15.220912605552897
A solution x is
[-1.10133381 -0.16360111 -0.89734939  0.03216603  0.6069123  -1.12687348
  1.12967856  0.88176638  0.49075229  0.8984822 ]
A dual solution is
[6.98804566e-10 6.11756416e-01 5.28171747e-01 1.07296862e+00
 3.93758849e-09 2.30153870e+00 4.25704004e-10 7.61206896e-01
 8.36905607e-09 2.49370377e-01 1.30187004e-09 2.06014070e+00
 3.22417207e-01 3.84054343e-01 1.59493641e-09]
"""