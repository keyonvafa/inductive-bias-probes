import argparse
import logging
import numpy as np
import tqdm
import yaml

from inductivebiasprobes.paths import GRIDWORLD_CONFIG_DIR, GRIDWORLD_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Gridworld white noise data")
    # Remove --num_states from arguments, as we will loop over 2,3,4,5
    parser.add_argument("--num_white_noise_datasets", type=int, default=100)
    parser.add_argument("--white_noise_dataset_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def generate_state_dependent_noise(
    num_states,
    states_train,
    states_val,
    random_seed,
):
    """Generate noise that is a deterministic function of the state.

    Args:
        num_states: Number of possible states
        states_train: Array of shape (train_dataset_size, sequence_length) containing state values for training
        states_val: Array of shape (val_dataset_size, sequence_length) containing state values for validation
        random_seed: Base random seed for reproducibility

    Returns:
        noise_train: Array of shape (train_dataset_size, sequence_length, 1) containing noise values for train set
        indices_train: Array of shape (train_dataset_size,) containing randomly sampled sequence indices for train set
        noise_val: Array of shape (val_dataset_size, sequence_length, 1) containing noise values for val set
        indices_val: Array of shape (val_dataset_size,) containing randomly sampled sequence indices for val set
    """
    random_state = np.random.RandomState(random_seed)

    train_dataset_size, seq_length, state_dim = states_train.shape
    val_dataset_size, _, _ = states_val.shape

    # Create a lookup table mapping each state to its noise value
    noise_lookup = {
        state: random_state.choice([0, 1]) for state in range(num_states)
    }
    noise_train = np.zeros((train_dataset_size, seq_length, 1))
    noise_val = np.zeros((val_dataset_size, seq_length, 1))
    for state, noise_value in noise_lookup.items():
        noise_train[states_train == state] = noise_value
        noise_val[states_val == state] = noise_value
    indices_train = random_state.randint(0, seq_length - 1, size=train_dataset_size)
    indices_val = random_state.randint(0, seq_length - 1, size=val_dataset_size)
    return noise_train, indices_train, noise_val, indices_val


def main():
    args = parse_args()
    random_state = np.random.RandomState(args.seed)

    for num_states in [2, 3, 4, 5]:
        logger.info(f"Processing num_states={num_states}")

        # Load the pre-generated gridworld data
        save_name = f"{num_states}-states"
        data_dir = GRIDWORLD_DATA_DIR / save_name
        config_dir = GRIDWORLD_CONFIG_DIR / save_name

        train_obs = np.load(data_dir / "obs_train.npy")
        val_obs = np.load(data_dir / "obs_val.npy")
        train_states = np.load(data_dir / "state_train.npy")
        val_states = np.load(data_dir / "state_val.npy")

        # Create white noise directory
        noise_data_dir = (
            data_dir
            / f"white_noise"
            / f"{args.white_noise_dataset_size}-examples"
        )
        noise_data_dir.mkdir(parents=True, exist_ok=True)

        # Sample validation indices once (reuse for all datasets)
        val_dataset_indices = random_state.choice(
            len(val_states), size=args.white_noise_dataset_size, replace=False
        )

        # Generate and save white noise datasets
        for dataset_idx in tqdm.trange(args.num_white_noise_datasets, desc=f"num_states={num_states}"):
            # Sample indices for this dataset
            train_dataset_indices = random_state.choice(
                len(train_states), size=args.white_noise_dataset_size, replace=False
            )

            # Get dataset observations and states
            train_dataset_obs = train_obs[train_dataset_indices]
            val_dataset_obs = val_obs[val_dataset_indices]
            train_dataset_states = train_states[train_dataset_indices]
            val_dataset_states = val_states[val_dataset_indices]

            # Generate noise for training and validation datasets
            train_noise, train_noise_indices, val_noise, val_noise_indices = (
                generate_state_dependent_noise(
                    num_states,
                    train_dataset_states,
                    val_dataset_states,
                    args.seed + dataset_idx * 100,
                )
            )

            # Save noise datasets and corresponding data
            np.save(
                noise_data_dir / f"white_noise_output_train_{dataset_idx}.npy",
                train_noise,
            )
            np.save(
                noise_data_dir / f"white_noise_output_val_{dataset_idx}.npy",
                val_noise,
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
                train_dataset_states,
            )
            np.save(
                noise_data_dir / f"white_noise_states_val_{dataset_idx}.npy",
                val_dataset_states,
            )
            np.save(
                noise_data_dir / f"white_noise_indices_train_{dataset_idx}.npy",
                train_noise_indices,
            )
            np.save(
                noise_data_dir / f"white_noise_indices_val_{dataset_idx}.npy",
                val_noise_indices,
            )

        # Save configs
        config = {
            "input_dim": 1,
            "output_dim": 1,
            "input_vocab_size": len(np.unique(train_obs)),
            "block_size": len(train_obs[0]) - 1,
            "mask_id": -1,
            "predict_type": f"white_noise",
            "output_vocab_size": 2,
            "num_data_points": args.white_noise_dataset_size,
        }
        config_name = f"white_noise_config.yaml"
        with open(config_dir / config_name, "w") as f:
            yaml.dump(config, f)

        logger.info(f"Saved white noise data to {noise_data_dir}")


if __name__ == "__main__":
    main()
