
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from inductivebiasprobes.paths import OTHELLO_CKPT_DIR

def load_metrics_from_ckpt(model_name):
    """
    Loads reconstruction metrics from the checkpoint directory for the given model.
    """
    ckpt_dir = (
        OTHELLO_CKPT_DIR
        / "synthetic_othello"
        / model_name
        / "next_token_pt_state_transfer"
        / "callback_results.json"
    )
    with open(ckpt_dir, "r") as f:
        data = json.load(f)
    steps = data["transfer_steps"]
    # Map the metrics to the ones shown in the image, in the correct order
    metrics = {
        "At least one move in predicted board is valid": data.get("one_move_in_common_frac", []),
        "All moves in predicted board are valid": data.get("reconstructed_is_subset_frac", []),
        "Set of valid moves match": data.get("next_move_frac", []),
        "Full board matches": data.get("same_board_frac", []),
    }
    return steps, metrics

def main():
    """
    Main function to generate plots for Mamba and Mamba-2, matching the style and legend of the provided image.
    """
    # Set seaborn style to match the image
    sns.set_theme(style="darkgrid")
    # Define colors to match the image as closely as possible
    color_map = {
        "At least one move in predicted board is valid": "#5b9bd5",  # blue
        "All moves in predicted board are valid": "#ed7d31",         # orange
        "Set of valid moves match": "#70ad47",                       # green
        "Full board matches": "#a94442",                             # red-brown
    }
    # Fallback to default color cycle if not enough colors
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))  # ~10.8x4.8/1.5 for compactness

    # Plot for Mamba
    mamba_steps, mamba_metrics = load_metrics_from_ckpt("mamba")
    for idx, (metric_name, values) in enumerate(mamba_metrics.items()):
        color = color_map.get(metric_name, default_colors[idx % len(default_colors)])
        ax1.plot(mamba_steps, values, label=metric_name, color=color, linewidth=2)
    ax1.set_xlabel("Transfer steps", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Mamba", fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5, alpha=0.7)

    # Plot for Mamba-2
    mamba2_steps, mamba2_metrics = load_metrics_from_ckpt("mamba2")
    for idx, (metric_name, values) in enumerate(mamba2_metrics.items()):
        color = color_map.get(metric_name, default_colors[idx % len(default_colors)])
        ax2.plot(mamba2_steps, values, label=metric_name, color=color, linewidth=2)
    ax2.set_xlabel("Transfer steps", fontsize=12)
    ax2.set_title("Mamba-2", fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5, alpha=0.7)

    # Remove y-label from right subplot for compactness
    ax2.set_ylabel("")

    # Create a single legend for the entire figure, matching the image
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=11,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    # Save to the experiments/othello directory for consistency
    output_dir = Path("figs")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "reconstruction_metrics_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()