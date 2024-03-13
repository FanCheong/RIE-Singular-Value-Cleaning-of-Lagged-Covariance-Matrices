import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess

figsize = (6, 4)          # Size of the figures
n = 3
T = 100

XT_matrix = np.zeros((n, T - 1))
XT_L1_matrix = np.zeros((n, T - 1))

phi_values = np.random.uniform(-0.9, 0.9, size=n)

sigma_epsilon_squared = 1
theoretical_list = [(phi * sigma_epsilon_squared) / (1 - phi**2) for phi in phi_values]
theoretical_list = np.array(sorted(theoretical_list, reverse=True))

for i, phi in enumerate(phi_values):
    ar1 = np.array([1, -phi])
    ma1 = np.array([1])
    AR_object = ArmaProcess(ar1, ma1)
    X = AR_object.generate_sample(nsample=T)
    print("X", X)
    
    XT_matrix[i, :] = X[1:]
    XT_L1_matrix[i, :] = X[:-1]
    # print(XT_matrix)
    # print(XT_L1_matrix)
    break
'''
Z = np.vstack([XT_matrix, XT_L1_matrix])
lagged_cov_matrix = np.asmatrix((Z @ Z.T) / (T - 1))

# Plot AR time series
fig, ax = plt.subplots(figsize=figsize)
for i in range(n):
    ax.plot(XT_matrix[i, :], label=f'$X_{i}$')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()
'''