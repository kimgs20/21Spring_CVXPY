# Portfolio optimization

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp

# a. Create an expected return vector and a covariance matrix
np.random.seed(1)
n = 10
mu = np.abs(np.random.randn(n, 1))
Sigma = np.random.randn(n, n)
Sigma = Sigma.T.dot(Sigma)

w = cp.Variable(n)
lamb = cp.Parameter(nonneg=True)
ret = mu.T @ w
risk = cp.quad_form(w, Sigma)

# b. Solve (MV.1)
prob = cp.Problem(cp.Maximize(ret - lamb * risk), [cp.sum(w) == 1, w >= 0])

SAMPLES = 100
risk_data = np.zeros(SAMPLES)
ret_data = np.zeros(SAMPLES)
lamb_vals = np.logspace(-2, 3, num=SAMPLES)  # increase exponentially

for i in range(SAMPLES):
    lamb.value = lamb_vals[i]
    prob.solve()
    risk_data[i] = cp.sqrt(risk).value
    ret_data[i] = ret.value

markers_on = [29, 40]  # ???
fig = plt.figure()
ax = fig.add_subplot(111)

# Draw line
plt.plot(risk_data, ret_data, 'g-')

# Draw blue square
for marker in markers_on:
    plt.plot(risk_data[marker], ret_data[marker], 'bs')  # blue sqaured
    ax.annotate(r"$\gamma = %.2f$" % lamb_vals[marker], xy=(risk_data[marker] + .08, ret_data[marker] - .03))

# Draw red circle
for i in range(n):
    plt.plot(cp.sqrt(Sigma[i, i]).value, mu[i], 'ro')
plt.xlabel('Standard deviaiton')
plt.ylabel('Return')
plt.show()
