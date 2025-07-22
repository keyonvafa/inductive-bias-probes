import argparse
import logging
import numpy as np
import yaml
from inductivebiasprobes.paths import GRIDWORLD_CONFIG_DIR, GRIDWORLD_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate gridworld data for multiple num_states settings")
    parser.add_argument("--sequence_length", type=int, default=100)
    parser.add_argument("--num_train_sequences", type=int, default=100_000)
    parser.add_argument("--num_valid_sequences", type=int, default=1_000)
    parser.add_argument("--allow_all_moves_at_boundary", action="store_true")
    parser.add_argument("--append_final_state", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def generate_sequences_vectorized(
    args, random_state, move_to_string, state_to_string, num_sequences, num_states
):
    """Generate sequences of moves and states for multiple trajectories in parallel.

    Args:
        args: Parsed command line arguments containing simulation parameters
        random_state: numpy RandomState object for reproducible random sampling
        move_to_string: Dictionary mapping move values (-1,0,1) to token indices
        state_to_string: Dictionary mapping state values to token indices
        num_sequences: Number of sequences to generate in parallel
        num_states: Number of states in the gridworld

    Returns:
        string_sequences: Array of shape (num_sequences, sequence_length) containing
            tokenized moves and optionally final states
        state_sequences: Array of shape (num_sequences, sequence_length) containing
            raw state values
    """
    # Initialize arrays for all sequences at once
    current_states = np.zeros((num_sequences,), dtype=np.int32)
    string_sequences = np.zeros((num_sequences, args.sequence_length), dtype=np.int32)
    state_sequences = np.zeros((num_sequences, args.sequence_length), dtype=np.int32)
    all_moves = np.array([-1, 0, 1])  # -1: left, 0: stay, 1: right

    # Generate moves for all sequences and steps at once
    for t in range(args.sequence_length):
        if args.allow_all_moves_at_boundary:
            # Broadcast allowed_moves for all sequences
            moves = random_state.choice(all_moves, size=num_sequences)
        else:
            # Create masks for different states
            at_zero = current_states == 0
            at_end = current_states == num_states - 1
            in_middle = ~(at_zero | at_end)

            # Generate moves based on position
            moves = np.zeros(num_sequences, dtype=np.int32)
            moves[at_zero] = random_state.choice([0, 1], size=np.sum(at_zero))
            moves[at_end] = random_state.choice([-1, 0], size=np.sum(at_end))
            moves[in_middle] = random_state.choice(all_moves, size=np.sum(in_middle))

        # Update states vectorized
        next_states = current_states + moves
        next_states = np.clip(next_states, 0, num_states - 1)
        current_states = next_states

        # Record sequences
        string_sequences[:, t] = np.vectorize(move_to_string.get)(moves)
        state_sequences[:, t] = current_states

    if args.append_final_state:
        string_sequences = np.pad(string_sequences, ((0, 0), (0, 1)))
        state_sequences = np.pad(state_sequences, ((0, 0), (0, 1)))
        string_sequences[:, -1] = np.vectorize(state_to_string.get)(current_states)
        state_sequences[:, -1] = current_states

    return np.atleast_3d(string_sequences), np.atleast_3d(state_sequences), random_state


def main():
    args = parse_args()
    random_state = np.random.RandomState(args.seed)

    # We'll generate data for num_states in [2, 3, 4, 5]
    for num_states in [2, 3, 4, 5]:
        logger.info(f"Generating data for num_states={num_states}")

        # Setup mappings
        move_to_string = {-1: 0, 0: 1, 1: 2}
        state_to_string = {i: i + 3 for i in range(num_states)}
        mask_id = -1

        # Generate all sequences at once using vectorized operations
        num_total = args.num_train_sequences + args.num_valid_sequences
        string_sequences, state_sequences, random_state = generate_sequences_vectorized(
            args, random_state, move_to_string, state_to_string, num_total, num_states
        )

        # Split into train/valid
        perm = random_state.permutation(num_total)
        train_data = string_sequences[perm[: args.num_train_sequences]]
        valid_data = string_sequences[perm[args.num_train_sequences :]]
        train_states = state_sequences[perm[: args.num_train_sequences]]
        valid_states = state_sequences[perm[args.num_train_sequences :]]

        # Setup save directory
        save_name = f"{num_states}-states"
        data_dir = GRIDWORLD_DATA_DIR / save_name
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save data files
        np.save(data_dir / "obs_train.npy", train_data.astype(np.uint8))
        np.save(data_dir / "obs_val.npy", valid_data.astype(np.uint8))
        np.save(data_dir / "state_train.npy", train_states.astype(np.uint8))
        np.save(data_dir / "state_val.npy", valid_states.astype(np.uint8))

        # Save config
        config_dir = GRIDWORLD_CONFIG_DIR / save_name
        config_dir.mkdir(parents=True, exist_ok=True)
        token_vocab_size = len(np.unique(train_data))
        common_config = {
            "input_dim": 1,
            "output_dim": 1,
            "input_vocab_size": token_vocab_size,
            "mask_id": mask_id,
            "block_size": len(train_data[0]) - 1,
            "num_data_points": args.num_train_sequences,
        }
        ntp_config = {
            **common_config,
            "output_vocab_size": token_vocab_size,
            "predict_type": "next_token",
        }
        state_config = {
            **common_config,
            "output_vocab_size": num_states,
            "predict_type": "state",
        }
        with open(config_dir / "ntp_config.yaml", "w") as f:
            yaml.dump(ntp_config, f)
        with open(config_dir / "state_config.yaml", "w") as f:
            yaml.dump(state_config, f)

        logger.info(f"Saved data to {data_dir}")
        logger.info(f"Saved config to {config_dir}")


if __name__ == "__main__":
    main()
