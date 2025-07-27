import argparse
import logging

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import torch.nn as nn
import torch.optim as optim
import tqdm
import yaml

from inductivebiasprobes.paths import PHYSICS_CONFIG_DIR, PHYSICS_DATA_DIR
from inductivebiasprobes.src.model import MLP
from inductivebiasprobes.src.train_utils import fit_mlp_full_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Physics white noise data")
    parser.add_argument("--num_white_noise_datasets", type=int, default=100)
    parser.add_argument("--white_noise_dataset_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--val_different_sequence",
        action="store_true",
        help="Use different sequence for val set",
    )
    parser.add_argument(
        "--num_projection_trials",
        type=int,
        default=50,
        help="Number of random projections to try for selecting the best one.",
    )
    return parser.parse_args()


def generate_linear_oracle_predictions(
    train_states,
    train_noise,
    val_states,
):
    state_dim, noise_dim = train_states.shape[-1], train_noise.shape[-1]
    linear_model = LinearRegression()
    linear_model.fit(
        train_states.reshape(-1, state_dim), train_noise.reshape(-1, noise_dim)
    )
    return linear_model.predict(val_states.reshape(-1, state_dim))


def generate_mlp_oracle_predictions(
    train_states,
    train_noise,
    val_states,
    hidden_dims=(5, 5),
    epochs=1_000,
    lr=5e-2,
):
    state_dim, noise_dim = train_states.shape[-1], train_noise.shape[-1]
    mlp = MLP(state_dim, hidden_dims, noise_dim)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    return (
        fit_mlp_full_batch(
            mlp,
            nn.MSELoss(),
            optimizer,
            train_states.reshape(-1, state_dim),
            train_noise.reshape(-1, noise_dim),
            val_states.reshape(-1, state_dim),
            epochs=epochs,
        )
        .cpu()
        .numpy()
    )


def generate_state_dependent_noise(
    states_train_std,
    states_val_std,
    random_seed,
    num_projection_trials=10,
):
    """Generate noise that is a deterministic function of the state.

    Args:
        states_train_std: Standardized training states, shape (dataset_size, sequence_length, state_dim)
        states_val_std: Standardized validation states, shape (dataset_size, sequence_length, state_dim)
        random_seed: Base random seed for reproducibility
        num_projection_trials: Number of random projections to try to find the best one.
    Returns:
        noise_train: Array of shape (dataset_size, sequence_length, 1) containing noise values for train set
        indices_train: Array of shape (dataset_size, 2) containing randomly sampled sequence indices for train set
        noise_val: Array of shape (dataset_size, sequence_length, 1) containing noise values for val set
        indices_val: Array of shape (dataset_size, 2) containing randomly sampled sequence indices for val set
    """
    local_random_state = np.random.RandomState(random_seed)

    train_dataset_size, seq_len, state_dim = states_train_std.shape
    val_dataset_size, _, _ = states_val_std.shape

    best_projection_matrix = None
    best_spearman_corr = -float("inf")

    # Prepare a subset of training data for evaluating projections
    # Reshape once before the loop
    states_train_flat_for_eval = states_train_std.reshape(-1, state_dim)
    # Further subset for pdist performance, using local_random_state for this choice
    subset_size_for_eval = min(1000, states_train_flat_for_eval.shape[0])
    subset_inds_for_eval = local_random_state.choice(
        len(states_train_flat_for_eval), size=subset_size_for_eval, replace=False
    )
    states_subset_for_eval = states_train_flat_for_eval[subset_inds_for_eval]
    original_dists_eval = pdist(states_subset_for_eval, metric="euclidean")

    logger.info(
        f"Evaluating {num_projection_trials} random projections to find the best one..."
    )
    for trial in range(num_projection_trials):
        # Generate a new random projection matrix using the local_random_state
        projection_matrix_trial = local_random_state.randn(state_dim, 1)

        # Project the subset
        projected_subset_eval = states_subset_for_eval @ projection_matrix_trial
        projected_dists_eval = pdist(
            projected_subset_eval.reshape(-1, 1), metric="euclidean"
        )

        if len(original_dists_eval) < 2 or len(projected_dists_eval) < 2:
            # spearmanr requires at least 2 data points
            current_spearman_corr = -float("inf")  # Or handle as an error/skip
        else:
            current_spearman_corr, _ = spearmanr(
                original_dists_eval, projected_dists_eval
            )

        if np.isnan(current_spearman_corr):
            current_spearman_corr = -float("inf")  # Treat NaN correlation as worst

        if current_spearman_corr > best_spearman_corr:
            best_spearman_corr = current_spearman_corr
            best_projection_matrix = projection_matrix_trial
            logger.debug(
                f"Trial {trial+1}/{num_projection_trials}: New best Spearman corr: {best_spearman_corr:.4f}"
            )

    if best_projection_matrix is None:
        logger.warning(
            "No suitable projection matrix found. Using a default random one."
        )
        # Fallback to a single random projection using the original random_seed logic if needed
        # This ensures projection_matrix is always defined.
        # Re-seed with the original random_seed for this fallback to be consistent with how it would be without trials
        fallback_random_state = np.random.RandomState(random_seed)
        best_projection_matrix = fallback_random_state.randn(state_dim, 1)
        best_spearman_corr = -1  # Placeholder

    logger.info(
        f"Selected best projection with Spearman correlation: {best_spearman_corr:.4f}"
    )

    # Use the best projection matrix for the final noise generation
    noise_train = states_train_std @ best_projection_matrix
    noise_val = states_val_std @ best_projection_matrix
    
    # --- Sanity Check with the chosen projection (optional, mostly for logging) ---
    # This part is similar to the loop's eval but on potentially different subset or for final verification
    # For consistency and to ensure we log stats for the chosen projection:
    states_flat_final_check = states_train_std.reshape(-1, state_dim)
    projected_flat_final_check = noise_train.reshape(
        -1, 1
    )  # Using noise_train which is based on best_projection_matrix

    subset_size_final = min(1000, len(states_flat_final_check))
    # Use local_random_state to pick subset to avoid impacting other random operations dependent on `random_seed`
    final_subset_inds = local_random_state.choice(
        len(states_flat_final_check), size=subset_size_final, replace=False
    )

    states_flat_final_subset = states_flat_final_check[final_subset_inds]
    projected_flat_final_subset = projected_flat_final_check[final_subset_inds]

    original_dists_final = pdist(states_flat_final_subset, metric="euclidean")
    projected_dists_final = pdist(
        projected_flat_final_subset.reshape(-1, 1), metric="euclidean"
    )

    if len(original_dists_final) >= 2 and len(projected_dists_final) >= 2:
        overall_corr, p_val = pearsonr(original_dists_final, projected_dists_final)
        spearman_final_corr, spearman_p_val = spearmanr(
            original_dists_final, projected_dists_final
        )
        logger.info(
            f"Final (chosen proj) Pearson correlation: {overall_corr:.4f} (p={p_val:.4f})"
        )
        logger.info(
            f"Final (chosen proj) Spearman correlation: {spearman_final_corr:.4f} (p={spearman_p_val:.4f})"
        )
    else:
        logger.warning(
            "Not enough data points in the final subset for correlation calculation."
        )
    # --- End Sanity Check ---
    # Sample random indices for each sequence in dataset
    # Only consider odd indices (e.g., 1, 3, 5, ...) because even corresponds to the sun
    # We use the original random_seed here, passed as an argument for the overall dataset generation consistency
    # The local_random_state was for the projection search only.
    main_random_state = np.random.RandomState(random_seed)
    indices_train = np.array(
        [
            2 * main_random_state.choice((seq_len - 1) // 2, size=2, replace=False) + 1
            for _ in range(train_dataset_size)
        ]
    )
    indices_val = np.array(
        [
            2 * main_random_state.choice((seq_len - 1) // 2, size=2, replace=False) + 1
            for _ in range(val_dataset_size)
        ]
    )
    return noise_train, indices_train, noise_val, indices_val


def main():
    args = parse_args()
    random_state = np.random.RandomState(args.seed)

    # Load the pre-generated physics data
    train_obs = np.load(PHYSICS_DATA_DIR / "obs_two_body_train.npy")
    train_states = np.load(PHYSICS_DATA_DIR / "state_two_body_train.npy")
    if args.val_different_sequence:
        val_obs = np.load(PHYSICS_DATA_DIR / "obs_two_body_val.npy")
        val_states = np.load(PHYSICS_DATA_DIR / "state_two_body_val.npy")
    else:
        # Extrapolate on the same sequence for val set
        val_obs = train_obs
        val_states = train_states

    # Get ntp_config
    ntp_config = yaml.load(
        open(PHYSICS_CONFIG_DIR / "ntp_config.yaml"), Loader=yaml.FullLoader
    )
    input_vocab_size = ntp_config["input_vocab_size"]

    # Create white noise directory
    noise_data_dir = (
        PHYSICS_DATA_DIR
        / "white_noise"
        / f"{args.white_noise_dataset_size}-examples"
    )
    noise_data_dir.mkdir(parents=True, exist_ok=True)

    # Fixed across datasets
    val_indices = random_state.choice(
        len(val_states), size=args.white_noise_dataset_size, replace=False
    )

    # Get mean and std of train states
    train_mean = train_states.mean(axis=(0, 1), keepdims=True)
    train_std = train_states.std(axis=(0, 1), keepdims=True)

    # Generate and save white noise datasets
    for dataset_idx in tqdm.trange(args.num_white_noise_datasets):
        # Sample indices for this dataset
        train_indices = random_state.choice(
            len(train_states), size=args.white_noise_dataset_size, replace=False
        )
        if not args.val_different_sequence:
            val_indices = train_indices

        # Get dataset observations and states
        train_dataset_obs = train_obs[train_indices]
        train_dataset_states = train_states[train_indices]
        val_dataset_obs = val_obs[val_indices]
        val_dataset_states = val_states[val_indices]

        # Standardize
        train_states_standardized = (train_dataset_states - train_mean) / train_std
        val_states_standardized = (val_dataset_states - train_mean) / train_std

        train_noise, train_noise_indices, val_noise, val_noise_indices = (
            generate_state_dependent_noise(
                train_states_standardized,
                val_states_standardized,
                args.seed
                + dataset_idx
                * 100,  # Different seed for each dataset noise generation process
                num_projection_trials=args.num_projection_trials,
            )
        )

        # Mask out the noise for the oracle prediction
        row_indices = np.arange(train_noise.shape[0])[:, np.newaxis]
        train_noise_masked = train_noise[row_indices, train_noise_indices]
        train_states_masked = train_states_standardized[
            row_indices, train_noise_indices
        ]

        # Linear oracle prediction
        logger.info("Generating linear oracle predictions...")
        val_linear_predicted_noise = generate_linear_oracle_predictions(
            train_states_masked,
            train_noise_masked,
            val_states_standardized,
        )
        rmse = np.sqrt(
            np.mean((val_linear_predicted_noise.ravel() - val_noise.ravel()) ** 2)
        )
        logger.info(f"RMSE of linear oracle predictions: {rmse:.4f}")

        # MLP oracle prediction
        logger.info("Generating MLP oracle predictions...")
        val_mlp_predicted_noise = generate_mlp_oracle_predictions(
            train_states_masked,
            train_noise_masked,
            val_states_standardized,
        )
        rmse = np.sqrt(
            np.mean((val_mlp_predicted_noise.ravel() - val_noise.ravel()) ** 2)
        )
        logger.info(f"RMSE of MLP oracle predictions: {rmse:.4f}")

        # Standardize noise
        train_noise_mean = train_noise.mean(axis=(0, 1), keepdims=True)
        train_noise_std = train_noise.std(axis=(0, 1), keepdims=True)

        train_noise_standardized = (train_noise - train_noise_mean) / train_noise_std
        val_noise_standardized = (val_noise - train_noise_mean) / train_noise_std
        val_linear_predicted_noise_standardized = (
            val_linear_predicted_noise - train_noise_mean
        ) / train_noise_std
        val_mlp_predicted_noise_standardized = (
            val_mlp_predicted_noise - train_noise_mean
        ) / train_noise_std

        # Save noise datasets and corresponding data
        np.save(
            noise_data_dir / f"white_noise_output_train_{dataset_idx}.npy",
            train_noise_standardized,
        )
        np.save(
            noise_data_dir / f"white_noise_output_val_{dataset_idx}.npy",
            val_noise_standardized,
        )
        np.save(
            noise_data_dir / f"white_noise_obs_train_{dataset_idx}.npy",
            train_dataset_obs,
        )
        np.save(
            noise_data_dir / f"white_noise_obs_val_{dataset_idx}.npy",
            val_dataset_obs,
        )
        np.save(
            noise_data_dir / f"white_noise_states_train_{dataset_idx}.npy",
            train_states_standardized,
        )
        np.save(
            noise_data_dir / f"white_noise_states_val_{dataset_idx}.npy",
            val_states_standardized,
        )
        np.save(
            noise_data_dir / f"white_noise_indices_train_{dataset_idx}.npy",
            train_noise_indices,
        )
        np.save(
            noise_data_dir / f"white_noise_indices_val_{dataset_idx}.npy",
            val_noise_indices,
        )
        np.save(
            noise_data_dir
            / f"white_noise_oracle_predictions_val_linear_{dataset_idx}.npy",
            val_linear_predicted_noise_standardized,
        )
        np.save(
            noise_data_dir
            / f"white_noise_oracle_predictions_val_mlp_{dataset_idx}.npy",
            val_mlp_predicted_noise_standardized,
        )

    # Save config
    _, seq_len, _ = train_obs.shape
    config = {
        "input_dim": 2,
        "block_size": seq_len - 1,
        "mask_id": -1,
        "output_dim": 1,
        "input_vocab_size": input_vocab_size,
        "output_vocab_size": None,
        "predict_type": "white_noise",
        "num_data_points": args.white_noise_dataset_size,
    }
    PHYSICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(PHYSICS_CONFIG_DIR / "white_noise_config.yaml", "w") as f:
        yaml.dump(config, f)

    logger.info(f"Saved white noise data to {noise_data_dir}")


if __name__ == "__main__":
    main()