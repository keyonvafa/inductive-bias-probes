import argparse
import json
import logging

import numpy as np
from scipy.spatial.distance import pdist

from inductivebiasprobes.paths import PHYSICS_EXT_DIR, PHYSICS_DATA_DIR
from inductivebiasprobes.src.plot_utils import generate_ib_plot_and_mae

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt",
        choices=["gpt", "mamba", "mamba2", "rnn", "lstm"],
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="next_token",
        choices=["scratch", "next_token", "state"],
    )
    parser.add_argument(
        "--white_noise_dataset_size",
        type=int,
        default=100,
        help="Number of training examples in white noise dataset",
    )
    parser.add_argument(
        "--num_white_noise_datasets",
        type=int,
        default=100,
        help="Number of white noise datasets",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2_000,
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=20,
        help="Number of bins for IB computation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)  # Set random seed for reproducibility

    all_extrapolations = []
    ext_dir = PHYSICS_EXT_DIR / "white_noise"
    ext_curr_dir = (
        ext_dir
        / args.model_type
        / f"pt_{args.pretrained}"
        / f"{args.white_noise_dataset_size}_examples"
        / f"{args.max_iters}_iters"
    )

    # Load states and raw trajectory data
    # states = (
    #     np.load(PHYSICS_EXT_DIR / "white_noise" / "states.npy").astype(np.int32)[:, :-1]
    # ).ravel()
    states = np.load(ext_dir / "states.npy")
    state_dim = states.shape[-1]
    states = states.reshape(-1, state_dim)

    noise_values, oracle_lin_preds, oracle_mlp_preds = [], [], []
    for dataset_idx in range(args.num_white_noise_datasets):
        # Load data
        extrapolations = np.load(
            ext_curr_dir / f"idx_{dataset_idx}" / "extrapolations.npy"
        )
        all_extrapolations.append(extrapolations.reshape(-1))
        # Load noise and oracle predictions
        noise = np.load(
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{args.white_noise_dataset_size}-examples"
            / f"white_noise_output_val_{dataset_idx}.npy"
        )
        oracle_lin_pred = np.load(
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{args.white_noise_dataset_size}-examples"
            / f"white_noise_oracle_predictions_val_linear_{dataset_idx}.npy"
        )
        oracle_mlp_pred = np.load(
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{args.white_noise_dataset_size}-examples"
            / f"white_noise_oracle_predictions_val_mlp_{dataset_idx}.npy"
        )
        noise_values.append(noise[:, :-1].reshape(-1))
        oracle_lin_preds.append(oracle_lin_pred[:, :-1].reshape(-1))
        oracle_mlp_preds.append(oracle_mlp_pred[:, :-1].reshape(-1))
    # Stack arrays from different seeds
    all_extrapolations = np.stack(
        all_extrapolations, axis=0
    )  # Shape: (num_datasets, num_total_examples)
    all_noise_values = np.stack(noise_values, axis=0)
    all_oracle_lin_preds = np.stack(oracle_lin_preds, axis=0)
    all_oracle_mlp_preds = np.stack(oracle_mlp_preds, axis=0)
    # Subsample random indices
    random_inds = np.random.choice(
        states.shape[0], size=args.num_examples, replace=False
    )
    # states_sample = states[random_inds]
    extrapolations_sample = all_extrapolations[:, random_inds]
    noise_values_sample = all_noise_values[:, random_inds]
    oracle_lin_preds_sample = all_oracle_lin_preds[:, random_inds]
    oracle_mlp_preds_sample = all_oracle_mlp_preds[:, random_inds]
    # state_dists = pdist(states_sample, metric="euclidean")
    all_extrap_dists = []
    all_noise_dists = []
    all_oracle_lin_dists = []
    all_oracle_mlp_dists = []
    for i in range(extrapolations_sample.shape[0]):
        extrap_dists = pdist(
            extrapolations_sample[i, :, np.newaxis], metric="euclidean"
        )
        all_extrap_dists.append(extrap_dists)
        noise_dists = pdist(noise_values_sample[i, :, np.newaxis], metric="euclidean")
        all_noise_dists.append(noise_dists)
        oracle_lin_dists = pdist(
            oracle_lin_preds_sample[i, :, np.newaxis], metric="euclidean"
        )
        all_oracle_lin_dists.append(oracle_lin_dists)
        oracle_mlp_dists = pdist(
            oracle_mlp_preds_sample[i, :, np.newaxis], metric="euclidean"
        )
        all_oracle_mlp_dists.append(oracle_mlp_dists)
    all_extrap_dists = np.stack(all_extrap_dists, axis=0)
    all_noise_dists = np.stack(all_noise_dists, axis=0)
    all_oracle_lin_dists = np.stack(all_oracle_lin_dists, axis=0)
    all_oracle_mlp_dists = np.stack(all_oracle_mlp_dists, axis=0)
    noise_dists = np.mean(all_noise_dists, axis=0)
    oracle_lin_dists = np.mean(all_oracle_lin_dists, axis=0)
    oracle_mlp_dists = np.mean(all_oracle_mlp_dists, axis=0)

    for oracle_name, oracle_dists in (
        ("noise", noise_dists),
        ("oracle_lin", oracle_lin_dists),
        ("oracle_mlp", oracle_mlp_dists),
    ):
        logger.info(f"Computing IB for {oracle_name}")
        bins = np.linspace(oracle_dists.min(), oracle_dists.max(), args.num_bins + 1)
        extrap_vals = []
        oracle_vals = []
        for i in range(len(bins) - 1):
            bin_lower = bins[i]
            bin_upper = bins[i + 1]
            extrap_dists_in_bin = extrap_dists[
                (oracle_dists > bin_lower) & (oracle_dists < bin_upper)
            ]
            oracle_dists_in_bin = oracle_dists[
                (oracle_dists > bin_lower) & (oracle_dists < bin_upper)
            ]
            extrap_vals.append(np.mean(extrap_dists_in_bin).item())
            oracle_vals.append(np.mean(oracle_dists_in_bin).item())

        # Save values
        with open(
            ext_dir
            / args.model_type
            / f"pt_{args.pretrained}"
            / f"{args.white_noise_dataset_size}_examples"
            / f"{args.max_iters}_iters"
            / f"ib_{oracle_name}_values.json",
            "w",
        ) as f:
            json.dump(
                {"extrap_vals": extrap_vals, "oracle_vals": oracle_vals}, f, indent=4
            )
    # Plot and save IB values
    file_dir = PHYSICS_EXT_DIR / "white_noise" / args.model_type
    generate_ib_plot_and_mae(
        "oracle_lin",
        file_dir,
        args.white_noise_dataset_size,
        args.max_iters,
        logger,
    )
    generate_ib_plot_and_mae(
        "oracle_mlp",
        file_dir,
        args.white_noise_dataset_size,
        args.max_iters,
        logger,
    )
    generate_ib_plot_and_mae(
        "noise",
        file_dir,
        args.white_noise_dataset_size,
        args.max_iters,
        logger,
    )


if __name__ == "__main__":
    main()
