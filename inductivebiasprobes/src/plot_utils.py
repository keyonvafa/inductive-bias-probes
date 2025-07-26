import json

import matplotlib.pyplot as plt
import numpy as np


def load_extrapolation_and_oracle_values(file_path):
    """Loads extrapolation and noise values from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["extrap_vals"], data["oracle_vals"]


def generate_ib_plot_and_mae(
    oracle_name, file_dir, white_noise_dataset_size, max_iters, logger
):
    # pt_types = ["scratch", "next_token"]
    pt_types = ["next_token"]
    files = {
        pt: file_dir
        / f"pt_{pt}"
        / f"{white_noise_dataset_size}_examples"
        / f"{max_iters}_iters"
        / f"ib_{oracle_name}_values.json"
        for pt in pt_types
    }

    # Load the data
    # scratch_extrap, oracle_vals = load_extrapolation_and_oracle_values(files["scratch"])
    ntp_extrap, oracle_vals = load_extrapolation_and_oracle_values(files["next_token"])

    # Prepare x-axis data (indices based on the length of one of the data arrays)
    x_values = np.arange(len(oracle_vals))

    # Create the plot
    plt.figure(figsize=(10, 7))  # Adjusted figure size for better readability
    plt.style.use("seaborn-v0_8-whitegrid")  # Using a seaborn style for aesthetics

    # Scatter plots for each dataset
    # plt.scatter(
    #     x_values, scratch_extrap, label="Extrapolations (scratch)", alpha=0.8, s=80
    # )
    plt.scatter(
        x_values, ntp_extrap, label="Extrapolations (NTP pretrained)", alpha=0.8, s=80
    )
    plt.scatter(x_values, oracle_vals, label=str(oracle_name), alpha=0.8, s=80)

    # Adding labels, title, and legend
    plt.xlabel("Bin Index")
    plt.ylabel("Distance Value")
    plt.title("Comparison of Model Extrapolations vs. Truth")
    plt.legend(loc="best")  # Automatically find the best location for the legend
    plt.grid(True)  # Ensure grid is visible

    # Save the plot
    plt.tight_layout()  # Adjust plot to prevent labels from being cut off
    plt.savefig(file_dir / f"ib_plot_{oracle_name}.png")
    plt.close()

    def calculate_mae(y_true, y_pred):
        """Calculates Mean Absolute Error."""
        return np.nanmean(np.abs(y_true - y_pred))

    # mae_scratch = calculate_mae(np.array(oracle_vals), np.array(scratch_extrap))
    mae_ntp = calculate_mae(np.array(oracle_vals), np.array(ntp_extrap))

    logger.info(f"MAE for {oracle_name}:")
    # logger.info(f"  Scratch: {mae_scratch:.4f}")
    logger.info(f"  NTP: {mae_ntp:.4f}")
    # Save the MAE values
    with open(file_dir / f"ib_mae_{oracle_name}.json", "w") as f:
        json.dump(
            # {"mae_scratch": mae_scratch, "mae_ntp": mae_ntp}, f
            {"mae_ntp": mae_ntp}, f
        )
