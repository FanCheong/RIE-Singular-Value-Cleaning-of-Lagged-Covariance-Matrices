import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Set the dimensions and lag order
n = 3  # Number of variables
p = 1  # Lag order

# Generate a random coefficient matrix
np.random.seed(42)  # Set seed for reproducibility
coefficients = np.random.uniform(-0.9, 0.9, size=(n, n))

# Set the error covariance matrix (replace this with your desired covariance matrix)
error_covariance = np.eye(n)  # Identity matrix for simplicity

# Set the number of time steps
timesteps = 100

# Generate the VAR process
var_process = sm.tsa.VAR(np.random.randn(timesteps, n))
var_process.coefs = coefficients

# Plot the VAR time series
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(range(timesteps), var_process.endog[:, i], label=f'Variable {i+1}')

plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
