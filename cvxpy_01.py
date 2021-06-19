# Simple optimization problem in CVXPY

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

# CVXPY relies on the open source solvers: ECOS, OSQP, SCS
# Default solver for QP is 'OSQP'
prob.solve()  # returns the optimal value
print("status:", prob.status)
print("optimal value(OSQP):", prob.value)
print("optimal var(OSQP):", x.value, y.value)

prob.solve(solver=cp.CVXOPT)
print("\nstatus:", prob.status)
print("optimal value(CVXOPT)", prob.value)
print("optimal var(CVXOPT)", x.value, y.value)

# Probelms are immutable: they cannot be changed after they are created.
# To change objective or constraints, create a new problem.

# Replace the objective
prob2 = cp.Problem(cp.Maximize(x + y), prob.constraints)
print("\noptimal value of prob2", prob2.solve(solver=cp.CVXOPT))  # prob.solve() return value of obj fn
print("optimal variables of prob2", x.value, y.value)

# Replace the constrain (x + y == 1)
constraints = [x + y <= 3] + prob2.constraints[1:]
prob3 = cp.Problem(prob2.objective, constraints)
print("\noptimal value of prob3", prob3.solve(cp.CVXOPT))
print("optimal variables of prob3", x.value, y.value)

"""
Result
status: optimal
optimal value(OSQP): 1.0
optimal var(OSQP): 1.0 1.570086213240983e-22

status: optimal
optimal value(CVXOPT) 0.9999999969933488
optimal var(CVXOPT) 0.9999999992483373 7.516628143199188e-10

optimal value of prob2 1.0000000000000002
optimal variables of prob2 1.4999999949276757 -0.4999999949276754

optimal value of prob3 2.999999982743166
optimal variables of prob3 2.3356052038019564 0.6643947789412092
"""
