"""
Created on Tue May  1 13:52:07 2018

@author: florent
@modified by: bryan
"""

from base_functions import *


# Global configuration settings
figsize = (6, 4)          # Size of the figures
do_legend = True          # Flag to control legend display in plots

do_save = False           # Flag to control whether to save figures
do_test = True            # Flag to control testing mode
factor_test = 10 if do_test else 100  # Factor to modify a value based on test mode

models = ['AR']           # List of models to be used
np.random.seed(42)        # Setting a seed for reproducibility of random operations


def Total_Cov(model, n, p, T):
    """
    Calculate the total covariance for different time series models.

    Parameters:
    model (str): Model type ('null_case_lagged', 'AR', or 'VAR').
    n (int): Number of variables in the model.
    p (int): Unused parameter.
    T (int): Number of time steps.

    Returns:
    tuple: Depending on the model, returns different structures containing theoretical values and covariance matrices.
    """
    if model not in models:
        raise ValueError(f"Model '{model}' is not supported. Supported models are {models}.")
    
    if model == 'null_case_lagged':
        return total_cov_null_case_lagged(n, T)
    elif model == 'AR':
        return total_cov_ar(n, T)
    elif model == 'VAR':
        return total_cov_var(n, T)


def histo(E, ax, color="b", reg_coeff=1, label="", linewidth=1):
    if label == "":
        ax.hist(
            E,
            bins=int(round(reg_coeff * len(E) ** 0.416 + 2)),
            density=True,
            histtype="step",
            color=color,
            linewidth=linewidth,
        )
    else:
        ax.hist(
            E,
            bins=int(round(reg_coeff * len(E) ** 0.416 + 2)),
            density=True,
            histtype="step",
            color=color,
            label=label,
            linewidth=linewidth,
        )


sv_labels = {'empirical': '$s_k$ (empirical sing. val.)',
             'cleaned': '$s_k^{\mathrm{cleaned}}$ (cleaned sing. val.)',
             'true': '$s_k^{\mathrm{true}}$ (true sing. val.)'}


def reconstruct_matrix(U, Clean_Sing_Val, V):
    S_cleaned_diag = np.diag(Clean_Sing_Val)
    
    reconstructed_matrix = U * S_cleaned_diag * V.T

    return reconstructed_matrix


def plot_results(my_model, True_s, Emp_Sing_Val, Clean_Sing_Val, RIE_flag, file_name,
                 dimension_details,
                 figsize=figsize):
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    # if my_model == 'null_case_lagged':
    #     ax.plot(Emp_Sing_Val, Clean_Sing_Val, "b.")
    #     plt.xlabel(sv_labels['empirical'], fontsize=15)
    #     plt.ylabel(sv_labels['cleaned'], fontsize=15)
    #     plt.tick_params(axis="both", which="major", labelsize=15)
    #     plt.xlim(left=0)
    #     if do_save:
    #         plt.savefig(f"{file_name}_{dimension_details}", bbox_inches="tight")
    #     plt.show()
    # else:
    #     if my_model == 'AR':
    #         x = np.linspace(min_s, max_s, Nlinspace)
    #         y = True_s
    #         ax.plot(
    #             x,
    #             y,
    #             color="r",
    #             linewidth=3,
    #             linestyle="--",
    #             label=sv_labels['true'],
    #         )
    histo(
        True_s,
        ax=ax,
        color="r",
        linewidth=3,
        label=sv_labels['true'],
    )
    histo(
        Emp_Sing_Val,
        ax=ax,
        color="k",
        linewidth=3,
        label=sv_labels['empirical'],
    )
    histo(
        Clean_Sing_Val,
        ax=ax,
        color="b",
        linewidth=3,
        label=sv_labels['cleaned'],
    )
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.yticks([])
    plt.legend(loc="upper right")
    plt.xlim(left=0)
    if do_save:
        plt.savefig(f"{file_name}_{dimension_details}", bbox_inches="tight")
    # plt.show()
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    linestyle = "-"
    fro_norm_empirical = np.sqrt(np.sum((np.array(True_s) - np.array(Emp_Sing_Val)) ** 2))
    fro_norm_cleaned = np.sqrt(np.sum((np.array(True_s) - np.array(Clean_Sing_Val)) ** 2))
    ax.plot(True_s, Emp_Sing_Val, linestyle=linestyle, color="k",
            label=f"{sv_labels['empirical']}, 'L2 Norm={fro_norm_empirical:.2f}")
    ax.plot(True_s, Clean_Sing_Val, linestyle=linestyle, color="b",
            label=f"{sv_labels['cleaned']}, 'L2 Norm={fro_norm_cleaned:.2f}")
    ax.plot(True_s, True_s, linestyle=linestyle, color="r",
            label=sv_labels['true'])
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.xlabel(sv_labels['true'], fontsize=15)
    plt.legend(loc="best")
    plt.ylim(bottom=0)
    if do_save:
        plt.savefig(file_name + f"_true_vs_emp_and_cleaned_{dimension_details}",
                    bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_matrix_heatmap(matrix, title="Matrix Heatmap"):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='coolwarm')
    fig.colorbar(cax)
    plt.title(title)
    plt.show()


def plot_difference_heatmap(matrix1, matrix2, title="Difference Heatmap"):
    difference = matrix1 - matrix2
    fig, ax = plt.subplots()
    cax = ax.matshow(difference, cmap='coolwarm')
    fig.colorbar(cax)
    plt.title(title)
    plt.show()
        

ns = ps =          [1, 10, 100, 1000]
Nlinspaces = Ts =  [2500, 2500, 2500, 2500]

fro_norms_true_vs_empirical = []
fro_norms_true_vs_cleaned = []
fro_norms_true_vs_isotonic_cleaned = []
fro_norms = []

for my_model in models:
    for n, p, T in zip(ns, ps, Ts):
        True_S, Etotale = Total_Cov(my_model, n=n, p=p, T=100)
        (s, U, V, new_s, new_s_isotonic) = RIE_Cross_Covariance(CZZemp=Etotale,
                                                            T=100,
                                                            n=n,
                                                            p=p,
                                                            return_all=True)

        _, _, emp_matrix = get_submatrices_of_full_cov_mat(CZZ=Etotale, n=n, p=p)
        true_matrix = np.diag(True_S)
        cleaned_matrix = reconstruct_matrix(U, new_s, V)
        isotonic_cleaned_matrix = reconstruct_matrix(U, new_s_isotonic, V)

        # Frobenius norm of the difference between the true and cleaned matrix
        fro_norm_true_vs_empirical = np.linalg.norm(true_matrix - emp_matrix, 'fro')
        fro_norm_true_vs_cleaned = np.linalg.norm(true_matrix - cleaned_matrix, 'fro')
        fro_norm_true_vs_isotonic_cleaned = np.linalg.norm(true_matrix - isotonic_cleaned_matrix, 'fro')

        print(f"Frobenius norm of the difference between the true and empirical matrix: {fro_norm_true_vs_empirical:.2f}")
        print(f"Frobenius norm of the difference between the true and cleaned matrix: {fro_norm_true_vs_cleaned:.2f}")
        print(f"Frobenius norm of the difference between the true and isotonic cleaned matrix: {fro_norm_true_vs_isotonic_cleaned:.2f}")

        fro_norms_true_vs_empirical.append(fro_norm_true_vs_empirical)
        fro_norms_true_vs_cleaned.append(fro_norm_true_vs_cleaned)
        fro_norms_true_vs_isotonic_cleaned.append(fro_norm_true_vs_isotonic_cleaned)

        plot_matrix_heatmap(true_matrix, "True Matrix")
        plot_matrix_heatmap(emp_matrix, "Empirical Matrix")
        plot_matrix_heatmap(cleaned_matrix, "Cleaned Matrix")
        plot_matrix_heatmap(isotonic_cleaned_matrix, "Isotonic Cleaned Matrix")

        plot_difference_heatmap(true_matrix, emp_matrix, "True vs. Empirical Matrix")
        plot_difference_heatmap(true_matrix, cleaned_matrix, "True vs. Cleaned Matrix")
        plot_difference_heatmap(true_matrix, isotonic_cleaned_matrix, "True vs. Isotonic Cleaned Matrix")

    fro_norms.append(fro_norms_true_vs_empirical)
    fro_norms.append(fro_norms_true_vs_cleaned)
    fro_norms.append(fro_norms_true_vs_isotonic_cleaned)
    fro_norms = np.array(fro_norms)

    # Plot a line graph of frobenius norm vs n, one line for empriical and one for cleaned and one for isotonic cleaned
    figsize = (6, 4)
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
    plt.ylabel("Frobenius Norm", fontsize=15)
    plt.legend(loc="best")
    plt.ylim(bottom=0)
    if do_save:
        plt.savefig(f"{my_model}_frobenius_norm_vs_n", bbox_inches="tight")
    plt.show()

    # Plot a heatmap of the frobenius norms
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    cax = ax.imshow(fro_norms, cmap='viridis', interpolation='nearest')

    cbar = fig.colorbar(cax)

    n_values = [str(n) for n in ns]
    stages = ['Empirical', 'Cleaned', 'Isotonic Cleaned']

    # X and Y axis labels
    ax.set_xticks(np.arange(len(n_values)))
    ax.set_xticklabels(n_values)
    ax.set_yticks(np.arange(len(stages)))
    ax.set_yticklabels(stages, rotation=45, ha="right")

    # Adding titles and labels
    ax.set_title('Frobenius Norm Differences Heatmap')
    ax.set_xlabel('Dimension (n)')
    ax.set_ylabel('Comparison Stages')

    plt.show()
    plt.close()

    # for n, p, T, Nlinspace in zip(ns, ps, Ts, Nlinspaces):
        
    #     True_s, Etotale = Total_Cov(my_model, n, p, T)
    #     print(len(True_s))
    #     print(Etotale.shape)
    #     (s, new_s, new_s_isotonic) = RIE_Cross_Covariance(CZZemp=Etotale,
    #                                                     T=T,
    #                                                     n=n,
    #                                                     p=p,
    #                                                     Return_Sing_Values_only=True)
    #     outdir = "data_and_figures/"
    #     outdir += sys.argv[0].split("/")[-1].replace(".py", "")
    #     outdir = os.path.join(outdir, f"Model={my_model}_Fixed_T_Vary_n_NormErr")
    #     dimension_details = "T=" + str(T) + "_n=" + str(n) + "_Q=" + str((T//n))
    #     outdir = os.path.join(outdir, dimension_details)
    #     os.makedirs(outdir, exist_ok=True)
    #     np.savetxt(outdir + "/theoretical_sing_val.txt", True_s)
    #     np.savetxt(outdir + "/empirical_sing_val.txt", s)
    #     np.savetxt(outdir + "/cleaned_sing_val.txt", new_s)
    #     np.savetxt(outdir + "/isotonic_cleaned_sing_val.txt", new_s_isotonic)
    #     for isotonic_flag in [0, 1]:
    #         print(isotonic_flag)
    #         print(my_model)
    #         the_new_s = new_s_isotonic if isotonic_flag else new_s
    #         # if isotonic_flag == 0:
    #         #     plot_results(
    #         #         my_model=my_model, True_s=True_s, Emp_Sing_Val=s, Clean_Sing_Val=the_new_s,
    #         #         RIE_flag=0,
    #         #         dimension_details=dimension_details,
    #         #         file_name=os.path.join(outdir, f"{my_model}_true_vs_emp_sing_val")
    #         #     )

    #         plot_results(
    #             my_model=my_model, True_s=True_s, Emp_Sing_Val=s, Clean_Sing_Val=the_new_s,
    #             RIE_flag=1,
    #             dimension_details=dimension_details,
    #             file_name=os.path.join(outdir, f"{my_model}_true_vs_emp_sing_val_vs_RIE_isotonic={isotonic_flag}")
    #         )