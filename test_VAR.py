import numpy as np
import matplotlib.pyplot as plt

# Set the dimensions and lag order
n = 3  # Number of variables
p = 1  # Lag order

# Generate coefficient matrix
np.random.seed(42)  # Set seed for reproducibility
coefficients = np.zeros((n,n))

for i in range(1,n):
    coefficients[i, i-1] = np.random.uniform(-0.9, 0.9)

# Set the error covariance matrix
error_covariance = np.eye(n)  # Identity matrix for simplicity

# Set the number of time steps
timesteps = 100

# Initialize the time series matrix
X = np.zeros((timesteps, n))

# Generate the VAR process
for t in range(p, timesteps):
    lagged_values = X[t-p:t, :].flatten()
    error_terms = np.random.multivariate_normal(np.zeros(n), error_covariance)
    X[t, :] = coefficients @ lagged_values + error_terms
    print(X.shape)

# Prepare the data
XT_matrix = X[1:, :]  # Current values (excluding the first row to align with lags)
XT_L1_matrix = X[:-1, :]  # Lagged values (excluding the last row)
print(XT_matrix.shape)
print(XT_L1_matrix.shape)

# Stack the matrices vertically for covariance calculation
Z_VAR = np.vstack([XT_matrix.T, XT_L1_matrix.T])  # Transpose to align variables as rows
print(Z_VAR.shape)

# Calculate the lagged covariance matrix
lagged_cov_matrix_VAR = np.cov(Z_VAR)
print(lagged_cov_matrix_VAR.shape)


# This gives you a covariance matrix where:
# - The top left quadrant is the covariance among the variables at time t,
# - The bottom right quadrant is the covariance among the variables at time t-1,
# - The off-diagonal quadrants represent the covariance between time t and t-1 variables.


# # Plot the time series
# for i in range(n):
#     plt.plot(range(timesteps), time_series[:, i], label=f'Variable {i+1}')

# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
