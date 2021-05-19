# simple optimization problem in CVXPY

import cvxpy as cp
from cvxpy import constraints

# Create two scalar optimization variables
x = cp.Variable()
y = cp.Variable()

# Create two constraints
constraints = [x + y == 1,
                x - y >= 1]

# Form objective
obj = cp.Minimize((x - y)**2)

# Form and solve problem
prob = cp.Problem(obj, constraints)
prob.solve()  # returns the optimal value
print("status: ", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)

"""
Result
status:  optimal
optimal value 1.0  # The value of obj fn. J(x, y) = 1
optimal var 1.0 1.570086213240983e-22  # x=1, y=0
"""