import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# a. Create an expected return vector(mu) and a covariance matrix(Sigma)
np.random.seed(1)
n = 10
mu = np.abs(np.random.randn(n))     # n risky assets with the expected returns
Sigma = np.random.randn(n, n)       # covariance of asset returns
Sigma = Sigma.T.dot(Sigma)
lamb = cp.Parameter(nonneg=True)    # risk preference parameter of investor, non-negative
iota = np.ones((n,1))               # vector of ones

# iota = np.ones((n,1))

# b. Solve (MV.1): QP
w = cp.Variable(n)      # 'investment proportion' or 'porfolio allocation vector'
ret = mu.T @ w
risk = cp.quad_form(w, Sigma)
prob = cp.Problem(cp.Minimize(risk - lamb * ret),
                 [iota.T @ w == 1, # equal to cp.sum(w) == 1
                  w >= 0])  # long position only (leverage is not mentioned)

# lamb_vals = np.logspace(-2, 3, num=SAMPLES)

# Compute trade-off curve
SAMPLES = 100
risk_data = np.zeros(SAMPLES)
ret_data = np.zeros(SAMPLES)
lamb_vals = np.logspace(-2, 3, num=SAMPLES)  # small risk preference

for i in range(SAMPLES):
    lamb.value = lamb_vals[i]
    prob.solve()  # Default solver: ECOS
    risk_data[i] = cp.sqrt(risk).value
    ret_data[i] = ret.value

# Plot trade-off curve
# a = np.linspace(0, 90, 10, dtype=int)
# markers_on = a.tolist()
markers_on = [30, 35, 40, 45, 50, 55, 60]

# markers_on = [30, 40, 50, 60, 70, 80, 90]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(risk_data, ret_data, 'g-')  # Efficient frontier

# plot lambda points
for marker in markers_on:
    plt.plot(risk_data[marker], ret_data[marker], 'bs')
    ax.annotate(r"$\lambda = %.2f$" % lamb_vals[marker], 
    xy=(risk_data[marker] + .08, ret_data[marker] - .03))

# plot indiviudal assets
for i in range(n):
    plt.plot(cp.sqrt(Sigma[i, i]).value, mu[i], 'ro')

plt.title('(MV.1)')
plt.xlabel('Risk (Standard deviation)')
plt.ylabel('Return')
plt.show()

print("\nThe optimal value is", prob.value)
print("A solution w is")
print(w.value)
