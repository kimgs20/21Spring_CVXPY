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
print()
"""
Result
status:  optimal
optimal value 1.0  # The value of obj fn. J(x, y) = 1
optimal var 1.0 1.570086213240983e-22  # x=1, y=0
"""

# Probelms are immutable: they cannot be changed after they are created.
# To change objective or constraints, create a new problem.

# Replace the objective
prob2 = cp.Problem(cp.Maximize(x + y), prob.constraints)
print("optimal value of prob2", prob2.solve())  

# value of prob.solve() is prob.value(optimal value of obj fn)


# Replace the constrain (x + y == 1)
constraints = [x + y <= 3] + prob2.constraints[1:]
prob3 = cp.Problem(prob2.objective, constraints)
print("optimal value of prob3", prob3.solve())
"""
Result
optimal value of prob2 0.9999999999945575
optimal value of prob3 2.9999999999746754
"""