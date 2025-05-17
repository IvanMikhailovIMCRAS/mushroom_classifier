import matplotlib.pyplot as plt
import numpy as np


def show_exp_var(exp_var: np.ndarray, output_file: str = ""):
    """Show explaned variation curve

    Args:
        exp_var (np.ndarray): explaned variation
        output_file (str, optional): name of output picture
    """

    cum_sum_eigenvalues = np.cumsum(exp_var)
    plt.bar(
        range(0, len(exp_var)),
        exp_var,
        alpha=0.5,
        align="center",
        label="Individual explained variance",
    )
    plt.step(
        range(0, len(cum_sum_eigenvalues)),
        cum_sum_eigenvalues,
        where="mid",
        label="Cumulative explained variance",
    )
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.legend(loc="best")
    plt.tight_layout()
    if output_file == "":
        plt.show()
    else:
        plt.savefig(fname=output_file, dpi=600)
    plt.close()
