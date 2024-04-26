"""
@author: bryan

This script contains a function to recreate the random phi values distribution
from the true singular values of an AR time series.

Done by calculating the phi values from the true singular values through the
quadratic formula.
"""


import numpy as np
from base_functions import *

def recreate_phis(gamma, sigma_epsilon_squared=1):
    # Recreate the phi values from the true singular values
    gamma_nonzero = np.where(gamma == 0, np.finfo(float).eps, gamma)
    discriminant = sigma_epsilon_squared ** 2 + 4 * gamma_nonzero ** 2
    phi_plus = (- sigma_epsilon_squared + np.sqrt(discriminant)) / (2 * gamma_nonzero)
    phi_minus = (- sigma_epsilon_squared - np.sqrt(discriminant)) / (2 * gamma_nonzero)
    return (phi_plus, phi_minus)


def compare_phis(original_phis, phi_plus, phi_minus):
    # Compare the original phi values with the recreated phi values
    for i in range(len(original_phis)):
        if np.isclose(original_phis[i], phi_plus[i], atol=1e-8):
            print(f"Original phi_{i} matches recreated phi_plus")
        elif np.isclose(original_phis[i], phi_minus[i], atol=1e-8):
            print(f"Original phi_{i} matches recreated phi_minus")
        else:
            print(f"Original phi_{i} does not match either recreated phi_plus or phi_minus")


def total_cov_ar_phi(n, T):
    """
    Calculate the total covariance for an AR time series.

    Parameters:
    n (int): Number of variables in the model.
    T (int): Number of time steps.

    Returns:
    tuple: A tuple containing a list of theoretical singular values and the lagged covariance matrix.
    """
    XT_matrix = np.zeros((n, T - 1))
    XT_L1_matrix = np.zeros((n, T - 1))

    phi_values = np.random.uniform(-0.5, 0.5, size=n)
    sigma_epsilon_squared = 1
    theoretical_list_AR = [(phi * sigma_epsilon_squared) / (1 - phi**2) for phi in phi_values]

    for i, phi in enumerate(phi_values):
        ar1 = np.array([1, -phi])
        ma1 = np.array([1])
        AR_object = ArmaProcess(ar1, ma1)
        X = AR_object.generate_sample(nsample=T)
        
        XT_matrix[i, :] = X[1:]
        XT_L1_matrix[i, :] = X[:-1]

    Z_AR = np.vstack([XT_matrix, XT_L1_matrix])
    lagged_cov_matrix_AR = np.asmatrix((Z_AR @ Z_AR.T) / (T - 1))

    return (phi_values, theoretical_list_AR, lagged_cov_matrix_AR)


# Set the parameters
n = 1250
T = 2500


# Test the function
phi_values, True_S, Etotale = total_cov_ar_phi(n, T)
phi_values = np.array(phi_values)
phi_plus, phi_minus = recreate_phis(True_S)
compare_phis(phi_values, phi_plus, phi_minus)
