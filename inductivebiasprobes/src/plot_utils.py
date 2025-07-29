import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_extrapolation_and_oracle_values(file_path):
    """Loads extrapolation and noise values from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["extrap_vals"], data["oracle_vals"]



def generate_ib_plot_and_mae(
    oracle_name, file_dir, white_noise_dataset_size, max_iters, logger
):
    """
    Generates a two-panel plot comparing model extrapolations to oracle values
    for both linear and MLP oracles, following the plotting style in the provided code.
    Also computes and saves MAE.
    """
    sns.set_theme("paper")
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Only using next_token for now, but can be extended
    pt_type = "next_token"

    # Load linear oracle data
    linreg_file = (
        file_dir
        / f"pt_{pt_type}"
        / f"{white_noise_dataset_size}_examples"
        / f"{max_iters}_iters"
        / "ib_oracle_lin_values.json"
    )
    mlp_file = (
        file_dir
        / f"pt_{pt_type}"
        / f"{white_noise_dataset_size}_examples"
        / f"{max_iters}_iters"
        / "ib_oracle_mlp_values.json"
    )

    # Load the data
    with open(linreg_file, "r") as f:
        linreg = json.load(f)
    with open(mlp_file, "r") as f:
        mlp = json.load(f)

    # For both, use oracle_vals as x, extrap_vals as y
    bins_linreg = np.array(linreg["oracle_vals"])
    gpt_linreg = np.array(linreg["extrap_vals"])
    bins_mlp = np.array(mlp["oracle_vals"])
    gpt_mlp = np.array(mlp["extrap_vals"])

    # Plotting
    fig = plt.figure(figsize=(10.8, 3.6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Linear oracle
    ax1.scatter(gpt_linreg, bins_linreg)
    ax1.plot(
        [min(bins_linreg), max(bins_linreg)],
        [min(bins_linreg), max(bins_linreg)],
        label='Matches oracle',
        linestyle='--',
        color='red'
    )
    ax1.set_xlabel('Transformer inductive bias', fontsize=14)
    ax1.set_ylabel('Oracle inductive bias', fontsize=14)
    ax1.set_title('Linear oracle', fontsize=14)
    ax1.legend(fontsize=14)

    # MLP oracle
    ax2.scatter(gpt_mlp, bins_mlp)
    ax2.plot(
        [min(bins_mlp), max(bins_mlp)],
        [min(bins_mlp), max(bins_mlp)],
        label='Matches oracle',
        linestyle='--',
        color='red'
    )
    ax2.set_xlabel('Transformer inductive bias', fontsize=14)
    ax2.set_ylabel('Oracle inductive bias', fontsize=14)
    ax2.set_title('MLP oracle', fontsize=14)
    ax2.legend(fontsize=14)

    plt.tight_layout()
    # Save the plot as PDF in the file_dir
    fig.savefig(file_dir / "oracle_calibration.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Calculate MAE for both oracles
    def calculate_mae(y_true, y_pred):
        return np.nanmean(np.abs(np.array(y_true) - np.array(y_pred)))

    mae_linreg = calculate_mae(bins_linreg, gpt_linreg)
    mae_mlp = calculate_mae(bins_mlp, gpt_mlp)

    logger.info(f"MAE for oracle_lin: {mae_linreg:.4f}")
    logger.info(f"MAE for oracle_mlp: {mae_mlp:.4f}")

    # Save the MAE values
    with open(file_dir / "ib_mae_oracle_lin.json", "w") as f:
        json.dump({"mae_ntp": mae_linreg}, f)
    with open(file_dir / "ib_mae_oracle_mlp.json", "w") as f:
        json.dump({"mae_ntp": mae_mlp}, f)
