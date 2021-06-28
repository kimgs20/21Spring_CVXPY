import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


# c. Solve (RO.1): QCQP
np.random.seed(1)
n = 10
mu = np.abs(np.random.randn(n))     # n risky assets with the expected returns
# mu_hat = np.abs(np.random.randn(n))  # empirical expected returns
Sigma = np.random.randn(n, n)       # covariance of asset returns
Sigma = Sigma.T.dot(Sigma)
Sigma_mu = (1/n)*np.diag(np.diag(Sigma))
lamb = cp.Parameter(nonneg=True)    # risk preference parameter of investor, non-negative
iota = np.ones((n,1))               # vector of ones
eps = cp.Parameter(nonneg=True)

w = cp.Variable(n)      # 'investment proportion' or 'porfolio allocation vector'
ret = mu.T @ w
risk = cp.quad_form(w, Sigma)
prob = cp.Problem(cp.Minimize(risk - lamb * ret),
                 [iota.T @ w == 1, # cp.sum(w) == 1,
                  cp.quad_form(w, Sigma_mu) <= eps])

# Compute trade-off curve
epsilons = [1, 2, 3]
SAMPLES = 100
risk_data = np.zeros((len(epsilons),SAMPLES))
ret_data = np.zeros((len(epsilons),SAMPLES))
lamb_vals = np.logspace(-2, 3, num=SAMPLES)

for k, epsilon in enumerate(epsilons):
    for i in range(SAMPLES):
        eps.value = epsilon
        lamb.value = lamb_vals[i]
        prob.solve()  # Default solver: ECOS
        risk_data[k, i] = cp.sqrt(risk).value
        ret_data[k, i] = ret.value

# Plot trade-off curve
fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(n):
    plt.plot(cp.sqrt(Sigma[i, i]).value, mu[i], 'ro')  # generated by randn

colors = ['r','g','b']
for k, epsilon in enumerate(epsilons):
    plt.plot(risk_data[k,:], ret_data[k,:], color=colors[k], label=r"$\epsilon$ = %d" % epsilon)

markers_on = [40, 50, 60, 70]
markers_color = ['rs','gs','bs']

# plot lambda points
for k in range(3):
    for marker in markers_on:
        plt.plot(risk_data[k, marker], ret_data[k, marker], markers_color[k])
        ax.annotate(r"$\lambda = %.2f$" % lamb_vals[marker], xy=(risk_data[k, marker] + .08, ret_data[k, marker] - .03))

plt.title('(RO.1)')
plt.xlabel('Risk (Standard deviation)')
plt.ylabel('Return')
plt.legend(loc='lower right')
plt.show()
