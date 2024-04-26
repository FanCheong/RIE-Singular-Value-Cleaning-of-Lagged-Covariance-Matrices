"""
Created on Tue May  1 13:52:07 2018

@author: florent
@modified by: bryan
"""

from base_functions import *


# Global configuration settings
figsize = (12, 8)                    # Size of the figures
do_save = 1                          # Flag to control whether to save figures
do_show = 0                          # Flag to control whether to show figures
do_test = 0                          # Flag to control testing mode
models = ['AR']   # List of models to be used
np.random.seed(42)                   # Setting a seed for reproducibility of random operations


# Number of variables and time steps
if do_test:
    ns = ps = [100, 1000]
    Ts = [2500, 2500]
else:
    ns = ps = [250, 500, 1250, 2500]
    Ts = [2500, 2500, 2500, 2500]


# Singular value labels
sv_labels = {'empirical': '$s_k^{\mathrm{empirical}}$ (Empirical Sing. Val.)',
            'cleaned': '$s_k^{\mathrm{cleaned}}$ (Cleaned Sing. Val.)',
            'isotonic_cleaned': '$s_k^{\mathrm{isotonic}}$ (Isotonic Cleaned Sing. Val.)',
            'true': '$s_k^{\mathrm{true}}$ (True Sing. Val.)',
            'general': '$s_k$ (Sing. Val.)'}


# Diagonal value labels
diag_labels = {'true': '$diag_k^{\mathrm{true}}$ (True Diag. Val.)',
                'empirical': '$diag_k^{\mathrm{empirical}}$ (Empirical Diag. Val.)',
                'cleaned': '$diag_k^{\mathrm{cleaned}}$ (Cleaned Diag. Val.)',
                'general': '$diag_k$ (Diag. Val.)'}


def Total_Cov(model, n, p, T):
    """
    Calculate the total covariance for different time series models.

    Parameters:
    model (str): Model type ('Random Gaussian', 'AR', or 'VAR').
    n (int): Number of variables in the model.
    p (int): Unused parameter.
    T (int): Number of time steps.

    Returns:
    tuple: Depending on the model, returns different structures containing theoretical values and covariance matrices.
    """
    if model not in models:
        raise ValueError(f"Model '{model}' is not supported. Supported models are {models}.")
    
    if model == 'Random Gaussian':
        return total_cov_random_gaussian(n, T)
    elif model == 'AR':
        return total_cov_ar(n, T)


def reconstruct_matrix(U, Clean_s, V):
    Diag_Clean_s = np.diag(Clean_s)

    assert U.shape[1] == Diag_Clean_s.shape[0] == V.shape[0]
    
    reconstructed_matrix = U * Diag_Clean_s * V

    return reconstructed_matrix


def show_save_close(my_model, title, dimension_details):
    if do_save:
        plt.savefig(f"{my_model}_{title}_{dimension_details}.png", bbox_inches="tight")
    if do_show:
        plt.show()
    plt.close()


def plot_line_graph(my_model, True_s, Emp_s, Clean_s,
                    dimension_details, title="Sing. Val. Line Graph", figsize=figsize):
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    linestyle = "-"

    True_s = np.sort(True_s)[::-1]
    Emp_s = np.sort(Emp_s)[::-1]
    Clean_s = np.sort(Clean_s)[::-1]
    
    ax.plot(True_s, Emp_s, linestyle=linestyle, color="k",
            label=sv_labels['empirical'])
    ax.plot(True_s, Clean_s, linestyle=linestyle, color="b",
            label=sv_labels['cleaned'])
    ax.plot(True_s, True_s, linestyle=linestyle, color="r",
            label=sv_labels['true'])
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xlabel(sv_labels['true'], fontsize=15)
    plt.ylabel(sv_labels['general'], fontsize=15)
    plt.ylim(bottom=0)
    plt.title(f"{my_model} {title} with {dimension_details}")
    plt.legend(loc="best")
    show_save_close(my_model, title, dimension_details)


def plot_line_graph_diag(my_model, True_d, Emp_d, Clean_d,
                dimension_details, title="Diag. Val. Line Graph", figsize=figsize):
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    linestyle = "-"

    True_d = np.sort(True_d)[::-1]
    Emp_d = np.sort(Emp_d)[::-1]
    Clean_d = np.sort(Clean_d)[::-1]

    ax.plot(True_d, Emp_d, linestyle=linestyle, color="k",
            label=diag_labels['empirical'])
    ax.plot(True_d, Clean_d, linestyle=linestyle, color="b",
            label=diag_labels['cleaned'])
    ax.plot(True_d, True_d, linestyle=linestyle, color="r",
            label=diag_labels['true'])
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.xlabel(diag_labels['true'], fontsize=18)
    plt.ylabel(diag_labels['general'], fontsize=18)
    plt.legend(loc="best")
    plt.ylim(bottom=0)
    plt.title(f"{my_model} {title} with {dimension_details}")
    show_save_close(my_model, title, dimension_details)


def plot_scatter(my_model, Emp_s, Clean_s, dimension_details, title="Empirical vs Cleaned Scatter Plot", figsize=figsize):
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.plot(Emp_s, Clean_s, "b.")
    plt.xlabel(sv_labels['empirical'], fontsize=15)
    plt.ylabel(sv_labels['cleaned'], fontsize=15)
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xlim(left=0)
    plt.title(f"{my_model} {title} with {dimension_details}")
    show_save_close(my_model, title, dimension_details)   


def plot_matrix_heatmaps(my_model, true_matrix, empirical_matrix, cleaned_matrix, dimension_details,
                        title="Heatmaps", figsize=figsize, vmin=0, vmax=0.05):
    
    if my_model == 'Random Gaussian':
        vmax = 0.01
    
    fig = plt.figure(figsize=(figsize), tight_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05]) 

    # Create the subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Plot empirical_matrix heatmap
    cax1 = ax1.imshow(empirical_matrix, cmap='Blues', vmin=vmin, vmax=vmax)
    ax1.set_title(f"Empirical Matrix")

    # Plot true_matrix heatmap
    cax2 = ax2.imshow(true_matrix, cmap='Blues', vmin=vmin, vmax=vmax)
    ax2.set_title(f"True Matrix")
    
    # Plot cleaned_matrix heatmap
    cax3 = ax3.imshow(cleaned_matrix, cmap='Blues', vmin=vmin, vmax=vmax)
    ax3.set_title(f"Cleaned Matrix") 

    # Create the colorbar
    cbar_ax = fig.add_subplot(gs[0, -1])
    fig.colorbar(cax3, cax=cbar_ax)

    plt.suptitle(f"{my_model} {title} with {dimension_details}", fontsize=18)

    show_save_close(my_model, title, dimension_details)


def plot_difference_heatmaps(my_model, true_matrix, empirical_matrix, cleaned_matrix, dimension_details,
                             title="Difference Heatmap", figsize=figsize, vmin=0, vmax=0.10):
    
    fig = plt.figure(figsize=(figsize), tight_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

    # Create the subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Calculate the difference between the matrices
    difference_empirical = np.abs(true_matrix - empirical_matrix)
    difference_cleaned = np.abs(true_matrix - cleaned_matrix)

    # Plot difference_empirical heatmap
    cax1 = ax1.imshow(difference_empirical, cmap='Blues', vmin=vmin, vmax=vmax)
    ax1.set_title(f"True vs Empirical Matrix")

    # Plot difference_cleaned heatmap
    cax2 = ax2.imshow(difference_cleaned, cmap='Blues', vmin=vmin, vmax=vmax)
    ax2.set_title(f"True vs Cleaned Matrix")

    # Create the colorbar
    cbar_ax = fig.add_subplot(gs[0, -1])
    fig.colorbar(cax2, cax=cbar_ax)

    plt.suptitle(f"{my_model} {title} with {dimension_details}", fontsize=18)

    show_save_close(my_model, title, dimension_details)


def plot_histogram(my_model, true_matrix, empirical_matrix, cleaned_matrix, dimension_details, title="Histogram", figsize=figsize):
    fig, axs = plt.subplots(1, 3, figsize=figsize, tight_layout=True)  # Create a 1x3 grid of subplots

    # Flatten the matrices for histogram plotting
    true_values = np.array(true_matrix).flatten()
    empirical_values = np.array(empirical_matrix).flatten()
    cleaned_values = np.array(cleaned_matrix).flatten()

    # Define the common range and bins based on the range of the empirical matrix values
    range_min = min(empirical_values)
    range_max = max(empirical_values)
    bins = np.linspace(range_min, range_max, 50)  # For example, 50 bins

    # Plot empirical_values histogram
    axs[0].hist(np.array(empirical_values).flatten(), bins=bins, color='blue', histtype='step', linewidth=3, density=True)
    axs[0].set_title(f"{my_model} Empirical Matrix")
    axs[0].set_ylabel('Frequency')
    axs[0].set_xlabel('Value')

    # Plot true_values histogram
    axs[1].hist(np.array(true_values).flatten(), bins=bins, color='blue', histtype='step', linewidth=3, density=True)
    axs[1].set_title(f"{my_model} True Matrix")
    axs[1].set_ylabel('Frequency')
    axs[1].set_xlabel('Value')

    # Plot cleaned_values histogram
    axs[2].hist(np.array(cleaned_values).flatten(), bins=bins, color='blue', histtype='step', linewidth=3, density=True)
    axs[2].set_title(f"{my_model} Cleaned Matrix")
    axs[2].set_ylabel('Frequency')
    axs[2].set_xlabel('Value')

    plt.suptitle(f"{my_model} {title} with {dimension_details}")

    show_save_close(my_model, title, dimension_details)


def plot_combined_histogram(my_model, true_matrix, empirical_matrix, cleaned_matrix, dimension_details, title="Combined Histogram", figsize=figsize):
    plt.figure(figsize=figsize, tight_layout=True)

    # Flatten the matrices for histogram plotting
    true_values = np.array(true_matrix).flatten()
    empirical_values = np.array(empirical_matrix).flatten()
    cleaned_values = np.array(cleaned_matrix).flatten()

    # Define the common range and bins based on the range of the empirical matrix values
    range_min = min(empirical_values)
    range_max = max(empirical_values)
    bins = np.linspace(range_min, range_max, 30)  # For example, 50 bins

    # Plot the histograms
    plt.hist(empirical_values, bins=bins, color='black', histtype='step', label='Empirical Matrix', linewidth=3, density=True)
    plt.hist(cleaned_values, bins=bins, color='blue', histtype='step', label='Cleaned Matrix', linewidth=3, density=True)
    plt.hist(true_values, bins=bins, color='red', histtype='step', label='True Matrix', linewidth=3, density=True, linestyle='--')
    

    plt.title(f"{my_model} {title} with {dimension_details}")
    plt.ylabel('Frequency')
    plt.xlabel('Value')
    plt.legend(loc='best')

    show_save_close(my_model, title, dimension_details)


def plot_error_line_graph(my_model, fro_norms_true_vs_empirical, fro_norms_true_vs_cleaned, fro_norms_true_vs_isotonic_cleaned,
                              ns, dimension_details="", title="Frobenius Norm vs n", ylabel="Frobenius Norm", figsize=figsize):
    # Plot a line graph of frobenius norm vs n, one line for empriical and one for cleaned and one for isotonic cleaned
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    linestyle = "-"
    ax.plot(ns, fro_norms_true_vs_empirical, linestyle=linestyle, color="k",
            label="Empirical")
    ax.plot(ns, fro_norms_true_vs_cleaned, linestyle=linestyle, color="b",
            label="Cleaned")
    ax.plot(ns, fro_norms_true_vs_isotonic_cleaned, linestyle=linestyle, color="g",
            label="Isotonic Cleaned")
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xlabel("n", fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.legend(loc="best")
    plt.ylim(bottom=0)
    plt.title(f"{my_model} {title} with {dimension_details}")

    show_save_close(my_model, title, dimension_details)


def plot_error_heatmap(my_model, fro_norms, ns, dimension_details="", title="Frobenius Norm Differences Heatmap", figsize=(12,14)):
    # Plot a heatmap of the frobenius norms
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    cax = ax.imshow(fro_norms.T, cmap='viridis', interpolation='nearest')

    fig.colorbar(cax)

    n_values = [str(n) for n in ns]
    stages = ['Empirical', 'Cleaned', 'Isotonic Cleaned']

    # X and Y axis labels
    ax.set_yticks(np.arange(len(n_values)))
    ax.set_yticklabels(n_values)
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha="right")

    # Adding titles and labels
    ax.set_title('Frobenius Norm Differences Heatmap')
    ax.set_ylabel('Dimension (n)')
    ax.set_xlabel('Comparison Stages')

    plt.suptitle(f"{my_model} {title} from {dimension_details}")

    show_save_close(my_model, title, dimension_details)


def calculate_signal_to_noise(matrix):
    # Extract diagonal and off-diagonal elements
    diag_elements = np.diag(matrix)
    off_diag_elemetns = matrix[~np.eye(matrix.shape[0], dtype=bool)].flatten()

    var_diag = np.var(diag_elements, ddof=1) / len(diag_elements) # ddof=1 for unbiased variance
    var_off_diag = np.var(off_diag_elemetns, ddof=1) / len(off_diag_elemetns)   # ddof=1 for unbiased variance

    return var_diag / var_off_diag


for my_model in models:
    # Lists to store the frobenius norms
    fro_norms_true_vs_empirical = []
    fro_norms_true_vs_cleaned = []
    fro_norms_true_vs_isotonic_cleaned = []
    fro_norms = []

    diag_diff_true_vs_empirical = []
    diag_diff_true_vs_cleaned = []
    diag_diff_true_vs_isotonic_cleaned = []

    SNR_emps = []
    SNR_cleaneds = []
    SNR_isotonic_cleaneds = []

    for n, p, T in zip(ns, ps, Ts):
        print(f"Model: {my_model}, n: {n}, T: {T}")

        # Generate the data and figures directory
        dimension_details = "T=" + str(T) + "_n=" + str(n)

        True_S, Etotale = Total_Cov(my_model, n=n, p=p, T=T)
        (s, U, V, new_s, new_s_isotonic, reconstructed_CXY) = RIE_Cross_Covariance(CZZemp=Etotale,
                                                            T=T,
                                                            n=n,
                                                            p=p,
                                                            return_all=True)

        CXXemp_matrix, _, CXYemp_matrix = get_submatrices_of_lagged_cov_mat(n=n, p=p, CZZ=Etotale)
        true_matrix = np.diag(True_S)
        cleaned_matrix = reconstruct_matrix(U, new_s, V)
        isotonic_cleaned_matrix = reconstruct_matrix(U, new_s_isotonic, V)

        assert np.allclose(cleaned_matrix, reconstructed_CXY)

        # Frobenius norm of the difference between the true and cleaned matrix
        fro_norm_true_vs_empirical = np.linalg.norm(true_matrix - CXYemp_matrix, 'fro')
        fro_norm_true_vs_cleaned = np.linalg.norm(true_matrix - cleaned_matrix, 'fro')
        fro_norm_true_vs_isotonic_cleaned = np.linalg.norm(true_matrix - isotonic_cleaned_matrix, 'fro')

        print(f"Frobenius norm of the difference between the true and empirical matrix: {fro_norm_true_vs_empirical:.5f}")
        print(f"Frobenius norm of the difference between the true and cleaned matrix: {fro_norm_true_vs_cleaned:.5f}")
        print(f"Frobenius norm of the difference between the true and isotonic cleaned matrix: {fro_norm_true_vs_isotonic_cleaned:.5f}")

        fro_norms_true_vs_empirical.append(fro_norm_true_vs_empirical)
        fro_norms_true_vs_cleaned.append(fro_norm_true_vs_cleaned)
        fro_norms_true_vs_isotonic_cleaned.append(fro_norm_true_vs_isotonic_cleaned)

        diag_elements_true = np.diag(true_matrix)
        diag_elements_empirical = np.diag(CXYemp_matrix)
        diag_elements_cleaned = np.diag(cleaned_matrix)
        diag_elements_isotonic_cleaned = np.diag(isotonic_cleaned_matrix)

        abs_diff_true_vs_empirical = np.abs(diag_elements_true - diag_elements_empirical)
        abs_diff_true_vs_cleaned = np.abs(diag_elements_true - diag_elements_cleaned)
        abs_diff_true_vs_isotonic_cleaned = np.abs(diag_elements_true - diag_elements_isotonic_cleaned)

        diag_diff_true_vs_empirical.append(np.sum(abs_diff_true_vs_empirical))
        diag_diff_true_vs_cleaned.append(np.sum(abs_diff_true_vs_cleaned))
        diag_diff_true_vs_isotonic_cleaned.append(np.sum(abs_diff_true_vs_isotonic_cleaned))

        print(f"Mean absolute difference between the true and empirical diagonal elements: {np.mean(abs_diff_true_vs_empirical):.5f}")
        print(f"Mean absolute difference between the true and cleaned diagonal elements: {np.mean(abs_diff_true_vs_cleaned):.5f}")
        print(f"Mean absolute difference between the true and isotonic cleaned diagonal elements: {np.mean(abs_diff_true_vs_isotonic_cleaned):.5f}")

        # Calculate the Signal to Noise Ratio
        SNR_emp = calculate_signal_to_noise(CXYemp_matrix)
        SNR_cleaned = calculate_signal_to_noise(cleaned_matrix)
        SNR_isotonic_cleaned = calculate_signal_to_noise(isotonic_cleaned_matrix)

        print(f"Signal to Noise Ratio for Empirical Matrix: {SNR_emp:.5f}")
        print(f"Signal to Noise Ratio for Cleaned Matrix: {SNR_cleaned:.5f}")
        print(f"Signal to Noise Ratio for Isotonic Cleaned Matrix: {SNR_isotonic_cleaned:.5f}")
    
        SNR_emps.append(SNR_emp)
        SNR_cleaneds.append(SNR_cleaned)
        SNR_isotonic_cleaneds.append(SNR_isotonic_cleaned)

        if my_model == 'AR':
            plot_line_graph_diag(my_model, diag_elements_true, diag_elements_empirical, diag_elements_cleaned, dimension_details)
            plot_line_graph(my_model, True_S, s, new_s, dimension_details)
    
        if my_model == 'Random Gaussian':
            plot_scatter(my_model, s, new_s, dimension_details)
            plot_scatter(my_model, s, new_s_isotonic, dimension_details, title="Empirical vs Isotonic Cleaned Scatter Plot")
        plot_matrix_heatmaps(my_model, true_matrix, CXYemp_matrix, cleaned_matrix, dimension_details)
        plot_difference_heatmaps(my_model, true_matrix, CXYemp_matrix, cleaned_matrix, dimension_details)
        plot_histogram(my_model, true_matrix, CXYemp_matrix, cleaned_matrix, dimension_details)
        plot_combined_histogram(my_model, true_matrix, CXYemp_matrix, cleaned_matrix, dimension_details)

    fro_norms.append(fro_norms_true_vs_empirical)
    fro_norms.append(fro_norms_true_vs_cleaned)
    fro_norms.append(fro_norms_true_vs_isotonic_cleaned)
    fro_norms = np.array(fro_norms)

    if my_model == 'AR':
        plot_error_line_graph(my_model, SNR_emps, SNR_cleaneds, SNR_isotonic_cleaneds, ns,
                              dimension_details=f"T= {str(T)} and from n= {str(ns[0])} to n= {str(ns[-1])}", title="SNR vs n", ylabel="Signal to Noise Ratio")
    plot_error_line_graph(my_model, diag_diff_true_vs_empirical, diag_diff_true_vs_cleaned, diag_diff_true_vs_isotonic_cleaned,
                                ns, dimension_details=f"T= {str(T)} and from n= {str(ns[0])} to n= {str(ns[-1])}", title="Diag. Val. vs n", ylabel="Mean Absolute Error")
    plot_error_line_graph(my_model, fro_norms_true_vs_empirical, fro_norms_true_vs_cleaned, fro_norms_true_vs_isotonic_cleaned,
                              ns, dimension_details=f"T= {str(T)} and from n= {str(ns[0])} to n= {str(ns[-1])}")

    plot_error_heatmap(my_model, fro_norms, ns, dimension_details=f"T= {str(T)} and from n= {str(ns[0])} to n= {str(ns[-1])}")
