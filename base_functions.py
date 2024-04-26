"""
Created on Thu Aug 16 17:32:18 2018

@author: florent
@modified by: bryan
"""


import os
import sys
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import quad
from sklearn.isotonic import IsotonicRegression
from statsmodels.tsa.arima_process import ArmaProcess


plt.rcParams['text.usetex'] = True
plt.rcParams["legend.fontsize"] = 15
plt.rcParams['axes.labelsize'] = 15


def total_cov_random_gaussian(n, T):
    """
    Calculate the total covariance for a random gaussian lagged time series.

    Parameters:
    n (int): Number of variables in the model.
    T (int): Number of time steps.

    Returns:
    tuple: A tuple containing a list of zeros (theoretical singular values) and the lagged covariance matrix.
    """
    # Generate the random Gaussian time series
    X = np.matrix(np.random.randn(n, T))

    XT = X[:, 1:]  # Exclude the first 1 columns
    XT_L1 = X[:, :-1]  # Exclude the last 1 columns

    Z = np.vstack([XT, XT_L1]) # Stack the matrices vertically for covariance calculation
    
    # Apply mean centering
    Z = Z - np.mean(Z, axis=1)
    lagged_cov_matrix = (Z @ Z.T) / (T - 1)

    return ([0] * n, lagged_cov_matrix)


def total_cov_ar(n, T):
    """
    Calculate the total covariance for an AR time series.

    Parameters:
    n (int): Number of variables in the model.
    T (int): Number of time steps.

    Returns:
    tuple: A tuple containing a list of theoretical singular values and the lagged covariance matrix.
    """
    # Initialize matrices
    XT_matrix = np.zeros((n, T))
    XT_L1_matrix = np.zeros((n, T))

    # Generate AR(1) coefficients
    phi_values = np.random.uniform(0.0, 0.9, size=n)
    sigma_epsilon_squared = 1
    
    # Calculate theoretical singular values for the diagonals
    true_s = [(phi * sigma_epsilon_squared / (1 - phi ** 2)) for phi in phi_values]

    # Simulate the AR(1) process
    for i, phi in enumerate(phi_values):
        ar1 = np.array([1, -phi])
        AR_object = ArmaProcess(ar1, [1])
        X = AR_object.generate_sample(nsample=T)
        
        XT_matrix[i, :] = X
        XT_L1_matrix[i, :] = np.roll(X, shift=1)  # Shift by tau
    
    # Stack original and lagged matrices
    Z = np.vstack([XT_matrix, XT_L1_matrix])

    # Apply mean centering
    Z = Z - Z.mean(axis=1, keepdims=True)

    # Calculate lagged covariance matrix
    lagged_cov_matrix = np.asmatrix((Z @ Z.T) / (T - 1))

    return (true_s, lagged_cov_matrix)


def check_matrix(M, n, p):
    assert M is not None
    assert isinstance(n, int)
    assert isinstance(p, int)
    assert isinstance(M, np.matrix), str(type(M))
    assert M.shape == (n, n), f"{M.shape}"


def get_submatrices_of_lagged_cov_mat(n, p, CZZ):
    check_matrix(CZZ, 2*n, 2*n)
    CXX = CZZ[np.ix_(range(n), range(n))]
    CYY = CZZ[np.ix_(range(n, 2*n), range(n, 2*n))]
    CXY = CZZ[np.ix_(range(n), range(n, 2*n))]
    return CXX, CYY, CXY


def Coeffs(n, p, U, V, CXXemp, CYYemp):
    for M in [U, CXXemp]:
        check_matrix(M, n, n)
    for M in [V, CYYemp]:
        check_matrix(M, n, n)
    Coeff_A, Coeff_B, Coeff_B_n_to_p = [], [], 0.0
    for k in range(n):
        Coeff_A.append((U[:, k].T * CXXemp * 
                        U[:, k]).item())  # Coefficients from X
        Coeff_B.append((V[k, :] * CYYemp * 
                        (V[k, :].T)).item()) # Coefficients from lagged X
    return Coeff_A, Coeff_B, Coeff_B_n_to_p


def approx_L_or_imLoimH(z,
                        n,
                        p,
                        T,
                        Coeff_A=None,
                        Coeff_B=None,
                        Coeff_B_n_to_p=None,
                        CXXemp=None,
                        CYYemp=None,
                        CXYemp=None,
                        U=None,
                        s=None,
                        stwo=None,
                        V=None,
                        algo_used=1,
                        return_L=False):
    assert algo_used in (1, 2), f"{algo_used}"
    assert isinstance(n, int)
    if algo_used == 1:
        if any([x is None for x in [Coeff_A, Coeff_B, Coeff_B_n_to_p]]):
            if U is None or V is None:
                check_matrix(CXYemp, n, n)
                U, s, V = np.linalg.svd(CXYemp, full_matrices=True)
            check_matrix(CXXemp, n, n)
            check_matrix(CYYemp, n, n)
            Coeff_A, Coeff_B, Coeff_B_n_to_p = Coeffs(n=n,
                                                      p=p,
                                                      U=U,
                                                      V=V,
                                                      CXXemp=CXXemp,
                                                      CYYemp=CYYemp)
        for c in (Coeff_A, Coeff_B):
            assert isinstance(c, list) and len(c) == n
        assert isinstance(Coeff_B_n_to_p, float), f"{Coeff_B_n_to_p}"
    ztwo = z ** 2
    if stwo is None:
        if s is None:
            check_matrix(CXYemp, n, n)
            s = np.linalg.svd(CXYemp, compute_uv=False)
        stwo = s ** 2
    assert isinstance(stwo, np.ndarray) and len(stwo) == n
    one_over_ztwo_minus_stwo = 1 / (ztwo - stwo)
    TH = np.dot(stwo, one_over_ztwo_minus_stwo)
    fT = float(T)
    H = TH / fT
    if algo_used == 1:
        L = 1 - T / (T + TH - (ztwo * np.dot(Coeff_A, one_over_ztwo_minus_stwo) * (
                np.dot(Coeff_B, one_over_ztwo_minus_stwo) + Coeff_B_n_to_p / ztwo) / (T + TH)))
    else:
        alpha, beta = n / fT, p / fT
        H = TH / fT
        G = (H + alpha) / ztwo
        L = (1 + 2 * H - np.sqrt(1 + 4 * ztwo * G * (G + (beta - alpha) / ztwo) * (1 + H) ** 2)) / (2 + 2 * H)
    return L if return_L else np.imag(L) / np.imag(H)


def RIE_Cross_Covariance(
        CZZemp,
        T,
        n,
        p,
        Return_Sing_Values_only=False,
        Return_Ancient_SV=False,
        Return_New_SV=False,
        Return_Sing_Vectors=False,
        adjust=False,
        return_all=False,
        isotonic=False,
        exponent_eta=0.5,
        c_eta=1,
        algo_used=1):
    """Flo's algo. We need n\leq p, Etotale needs to be of the type np.matrix"""
    assert algo_used in (1, 2), f"{algo_used}"
    CXXemp, CYYemp, CXYemp = get_submatrices_of_lagged_cov_mat(n=n, p=p, CZZ=CZZemp)
    U, s, V = np.linalg.svd(CXYemp, full_matrices=True)
    if algo_used == 1:
        Coeff_A, Coeff_B, Coeff_B_n_to_p = Coeffs(n=n,
                                                  p=p,
                                                  U=U,
                                                  V=V,
                                                  CXXemp=CXXemp,
                                                  CYYemp=CYYemp)
    else:
        Coeff_A, Coeff_B, Coeff_B_n_to_p = None, None, None
    stwo = s ** 2
    eta = c_eta * (n ** 2 * T) ** (-exponent_eta / 3.0)
    new_s = [max(0, approx_L_or_imLoimH(z=s[k] + 1j * eta,
                                        n=n,
                                        p=p,
                                        T=T,
                                        Coeff_A=Coeff_A,
                                        Coeff_B=Coeff_B,
                                        Coeff_B_n_to_p=Coeff_B_n_to_p,
                                        CXXemp=CXXemp,
                                        CYYemp=CYYemp,
                                        CXYemp=CXYemp,
                                        U=U,
                                        V=V,
                                        stwo=stwo,
                                        algo_used=algo_used,
                                        return_L=False)) * s[k] for k in range(n)]
    if adjust:
        new_s = np.array(new_s)
        new_s *= np.sqrt(
            max(0, (T * sum(stwo) - np.trace(CXXemp) * np.trace(CYYemp)) / (T + 1 - 2 * T ** (-1)))) / np.linalg.norm(
            new_s)
    if Return_Sing_Values_only:
        ir = IsotonicRegression()
        new_s_isotonic = ir.fit_transform(np.arange(n)[::-1], np.array(new_s))
        return (s, new_s, new_s_isotonic)
    if return_all:
        ir = IsotonicRegression()
        new_s_isotonic = ir.fit_transform(np.arange(n)[::-1], np.array(new_s))
        U, V = np.matrix(U), np.matrix(V)
        reconstructed_CXY = U * np.diag(new_s) * V
        return s, U, V, new_s, new_s_isotonic, reconstructed_CXY
    if isotonic:
        ir = IsotonicRegression()
        new_s = ir.fit_transform(np.arange(n)[::-1], np.array(new_s))
    U, V = np.matrix(U), np.matrix(V)
    res = [U * np.diag(new_s) * V]
    if Return_Ancient_SV:
        res.append(s)
    if Return_New_SV:
        res.append(new_s)
    if Return_Sing_Vectors:
        res.append(U)
        res.append(V)
    if Return_Ancient_SV or Return_New_SV or Return_Sing_Vectors:
        return tuple(res)
    else:
        print("Result shape in RIE CC function", res[0].shape)
        return res[0]
