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

# Set the error covariance matrix (replace this with your desired covariance matrix)
error_covariance = np.eye(n)  # Identity matrix for simplicity

# Set the number of time steps
timesteps = 100

# Initialize the time series matrix
time_series = np.zeros((timesteps, n))

# Generate the VAR process
for t in range(p, timesteps):
    lagged_values = time_series[t-p:t, :].flatten()
    error_terms = np.random.multivariate_normal(np.zeros(n), error_covariance)
    time_series[t, :] = coefficients.dot(lagged_values) + error_terms

# Plot the time series
for i in range(n):
    plt.plot(range(timesteps), time_series[:, i], label=f'Variable {i+1}')

plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
