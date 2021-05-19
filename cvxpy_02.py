# Infeasible and unbounded problems

import cvxpy as cp

x = cp.Variable()

# An infeasible problem
prob = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)

# An unbounded problem
prob = cp.Problem(cp.Minimize(x))
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)

"""
Result
status: infeasible
optimal value inf
status: unbounded
optimal value -inf
"""

# Notice
'''
In minimization problem,
the optimal value is 'inf' if infeasible
and '-inf' if unbounded

In maximization problem, the oppposite is true.
'''