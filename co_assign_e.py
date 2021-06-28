import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# a. Create an expected return vector(mu) and a covariance matrix(Sigma)
np.random.seed(1)
n = 10
mu = np.abs(np.random.randn(n))     # n risky assets with the expected returns
Sigma = np.random.randn(n, n)       # covariance of asset returns
Sigma = Sigma.T.dot(Sigma)
Sigma_mu = (1/n)*np.diag(np.diag(Sigma))
lamb = cp.Parameter(nonneg=True)    # risk preference parameter of investor, non-negative
iota = np.ones((n,1))               # vector of ones
eps = cp.Parameter(nonneg=True)
key = cp.Parameter(nonneg=True)

w = cp.Variable(n)      # 'investment proportion' or 'porfolio allocation vector'
W = cp.Variable((n, n), PSD=True)

ret_b = mu.T @ w
risk_b = cp.quad_form(w, Sigma)

ret_c = mu.T @ w
risk_c = cp.quad_form(w, Sigma)

ret_d = iota.T @ W @ mu
risk_d = cp.trace(Sigma @ W)

prob_b = cp.Problem(cp.Minimize(risk_b - lamb * ret_b),
                 [iota.T @ w == 1])       # [cp.sum(w) == 1,
                #   w >= 0])  # long position only (leverage is not mentioned)

prob_c = cp.Problem(cp.Minimize(risk_c - lamb * ret_c),
                 [iota.T @ w == 1, # cp.sum(w) == 1,
                  cp.quad_form(w, Sigma_mu) <= eps])

prob_d = cp.Problem(cp.Minimize(risk_d - lamb * ret_d),
                 [cp.trace(iota @ iota.T @ W) == 1,
                  cp.trace(Sigma_mu @ W) <= eps,
                  iota.T @ cp.abs(W) @ iota <= key*cp.trace(W)])

# Compute trade-off curve
epsilons = [1, 2, 3]
kays = [3, 5, 7]
# kays = [1, 2, 3]


SAMPLES = 100
w_vec = np.zeros((n,SAMPLES))
risk_data_b = np.zeros(SAMPLES)
ret_data_b = np.zeros(SAMPLES)
risk_data_c = np.zeros((len(epsilons),SAMPLES))
ret_data_c = np.zeros((len(epsilons),SAMPLES))
risk_data_d = np.zeros((len(kays),SAMPLES))
ret_data_d = np.zeros((len(kays),SAMPLES))
lamb_vals = np.logspace(-2, 3, num=SAMPLES)

for i in range(SAMPLES):
    lamb.value = lamb_vals[i]
    prob_b.solve()  # Default solver: ECOS
    risk_data_b[i] = cp.sqrt(risk_b).value
    ret_data_b[i] = ret_b.value

for k, epsilon in enumerate(epsilons):
    for i in range(SAMPLES):
        eps.value = epsilon
        lamb.value = lamb_vals[i]
        prob_c.solve()  # Default solver: ECOS
        risk_data_c[k, i] = cp.sqrt(risk_c).value
        ret_data_c[k, i] = ret_c.value

for idx, kay in enumerate(kays):
    for i in range(SAMPLES):
        eps.value = 1
        key.value = kays[idx]
        lamb.value = lamb_vals[i]
        prob_d.solve()
        eigval, eigvec = np.linalg.eig(W.value)
        max_idx = np.argmax(eigval)
        w_vec[:,i] = eigvec[:, max_idx]
        w_vec[:,i] = w_vec[:,i]/sum(w_vec[:,i])
        risk_data_d[idx, i] = cp.sqrt(cp.quad_form(w_vec[:,i], Sigma).value).value
        ret_data_d[idx, i] = mu.T @ w_vec[:,i]

# Plot trade-off curve
markers_on = [40, 50, 60, 70]
markers_color = ['rs','gs','bs']

fig = plt.figure()
ax = fig.add_subplot(111)

# plot curve for b
plt.plot(risk_data_b, ret_data_b, 'k-', label="(MV.1)")  # Efficient frontier

# plot lambda points for b
# for marker in markers_on:
#     plt.plot(risk_data_b[marker], ret_data_b[marker], 'ks')
#     ax.annotate(r"$\lambda = %.2f$" % lamb_vals[marker], xy=(risk_data_b[marker] + .08, ret_data_b[marker] - .03))

# plot indiviudal assets
for i in range(n):
    plt.plot(cp.sqrt(Sigma[i, i]).value, mu[i], 'ro')  # generated by randn

# plot curves for c
colors = ['r','g','b']
for k, epsilon in enumerate(epsilons):
    plt.plot(risk_data_c[k,:], ret_data_c[k,:], color=colors[k], label=r"(RO.1) $\epsilon$ = %d" % epsilon)

# plot lambda points for c
# for k in range(3):
#     for marker in markers_on:
#         plt.plot(risk_data_c[k, marker], ret_data_c[k, marker], markers_color[k])
#         ax.annotate(r"$\lambda = %.2f$" % lamb_vals[marker], xy=(risk_data_c[k, marker] + .08, ret_data_c[k, marker] - .03))

# plot curves for d
for idx, kay in enumerate(kays):
    plt.plot(risk_data_d[idx,:], ret_data_d[idx,:], linestyle='dashed', color=colors[idx], label=r"(RO.4) $k$ = %d, $\epsilon$ = 1" % kay)

# plot lambda points for d
# for idx in range(3):
#     for marker in markers_on:
#         plt.plot(risk_data_d[idx, marker], ret_data_d[idx, marker], markers_color[idx])
#         ax.annotate(r"$\lambda = %.2f$" % lamb_vals[marker], xy=(risk_data_d[idx, marker] + .08, ret_data_d[idx, marker] - .03))



plt.title('All trade-off curves')
plt.xlabel('Risk (Standard deviation)')
plt.ylabel('Return')
plt.xlim([0, 7])
plt.ylim([0, 6])
plt.legend(loc='lower right')
plt.show()
