import argparse
import collections
import logging
import numpy as np
import operator

from inductivebiasprobes import ReversibleOthelloBoardState
import tqdm
import yaml

from inductivebiasprobes.paths import OTHELLO_CONFIG_DIR, OTHELLO_DATA_DIR
from inductivebiasprobes.experiments.othello.generate_data import (
    get_word_to_id_mapping,
    game_to_states,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Othello white noise data")
    parser.add_argument("--num_white_noise_datasets", type=int, default=100)
    parser.add_argument("--white_noise_dataset_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def generate_state_dependent_noise(
    states, random_seed, state_to_noise=None
):
    """Generate noise that is a deterministic function of the state.

    Args:
        states: Array of shape (batch_size, sequence_length, 64) containing state values
        random_seed: Base random seed for reproducibility
        state_to_noise: Dictionary mapping states to noise values

    Returns:
        noise: Array of shape (batch_size, sequence_length, 1) containing noise values
        indices: Array of shape (batch_size,) containing randomly sampled sequence indices
        state_to_noise: Updated dictionary mapping states to noise values
    """
    batch_size, seq_length, state_dim = states.shape
    random_state = np.random.RandomState(random_seed)

    # Dictionary to store state -> noise bit mapping
    if state_to_noise is None:
        state_to_noise = {}

    # Reshape states to 2D for easier processing
    states_2d = states.reshape(-1, state_dim)
    noise_flat = np.zeros((states_2d.shape[0], 1))

    # Assign/lookup noise bit for each unique state
    for i, state in enumerate(states_2d):
        state_tuple = tuple(state)
        if state_tuple not in state_to_noise:
            # New state: assign random bit (0/1)
            state_to_noise[state_tuple] = random_state.randint(0, 2)
        noise_flat[i] = state_to_noise[state_tuple]

    # Reshape to original dimensions
    noise = noise_flat.reshape(batch_size, seq_length, 1)

    # Sample random indices for each sequence in batch
    indices = random_state.randint(0, seq_length - 1, size=batch_size)

    return noise.astype(np.float32), indices, state_to_noise


def generate_validation_set(depth=10):
    """Generate validation set with sequences that share same tokens but lead to different boards.

    Args:
        depth: Length of sequences to generate

    Returns:
        val_obs: Array of shape (num_sequences, 60) containing observation sequences
        val_states: Array of shape (num_sequences, 60, 64) containing state sequences
    """

    # Get all length-8 sequences and boards
    def explore_moves(game, depth, max_depth, moves_so_far, all_sequences, all_boards):
        if depth == max_depth:
            all_sequences.append(list(moves_so_far))
            all_boards.append(game.get_state())
            return

        valid_moves = game.get_valid_moves()
        for move in valid_moves:
            try:
                flip_info = game.get_flips_info(move)
            except ValueError:
                # Illegal move
                continue

            # Apply current move
            game.apply_move(flip_info)
            moves_so_far.append(move)

            # Recurse
            explore_moves(
                game, depth + 1, max_depth, moves_so_far, all_sequences, all_boards
            )

            # Undo
            moves_so_far.pop()
            game.undo_move(flip_info)

    all_sequences = []
    all_boards = []
    initial_game = ReversibleOthelloBoardState()
    logger.info("Generating validation set...")
    import time

    time_start = time.time()
    explore_moves(initial_game, 0, depth, [], all_sequences, all_boards)
    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start} seconds")

    # Map each board to a unique ID
    all_boards = np.array(all_boards)
    unique_boards = {}
    board_ids = []
    for board in all_boards:
        board_tuple = tuple(board.flatten())
        if board_tuple not in unique_boards:
            unique_boards[board_tuple] = len(unique_boards)
        board_ids.append(unique_boards[board_tuple])

    board_ids = np.array(board_ids)

    # Get a common board with many permutations
    same_tokens_indices = []
    idx = 5
    most_common_board = np.argsort(np.bincount(board_ids))[-idx]
    most_common_board_indices = np.where(board_ids == most_common_board)[0]
    most_common_board_sequences = [all_sequences[i] for i in most_common_board_indices]
    board_set = set(most_common_board_sequences[0])
    same_tokens_indices.extend(
        [i for i, x in enumerate(all_sequences) if set(x) == board_set]
    )

    # Convert sequences to observations and states
    word_to_id, pad_id = get_word_to_id_mapping()
    valid_data = pad_id * np.ones((len(same_tokens_indices), 60), dtype=np.int8)
    valid_states = []
    for i, index in enumerate(same_tokens_indices):
        seq = all_sequences[index]
        for j, word in enumerate(seq):
            valid_data[i, j] = word_to_id.get(str(word), pad_id)
        valid_states.append(game_to_states(seq))
    valid_states = np.array(valid_states)

    # Get uniqueness stats
    states_2d = valid_states.reshape(-1, 64)
    states_tuple = tuple(map(tuple, states_2d))
    state_counts = collections.Counter(states_tuple)
    duplicate_states = {
        state: count for state, count in state_counts.items() if count > 1
    }
    sorted_duplicates = sorted(
        duplicate_states.items(), key=operator.itemgetter(1), reverse=True
    )
    logger.info(f"Total states: {states_2d.shape[0]}")
    logger.info(f"Unique states: {len(state_counts)}")
    logger.info("\nStates appearing multiple times (showing top 10):")
    for state, count in sorted_duplicates[:10]:
        logger.info(f"Count: {count}, State: {np.array(state)}")

    return np.atleast_3d(valid_data), np.atleast_3d(valid_states)


def main():
    args = parse_args()
    random_state = np.random.RandomState(args.seed)

    # Load the pre-generated Othello data
    data_name = "synthetic_othello"
    data_dir = OTHELLO_DATA_DIR / data_name
    config_dir = OTHELLO_CONFIG_DIR / data_name
    config_dir.mkdir(parents=True, exist_ok=True)

    train_obs = np.load(data_dir / "obs_train.npy")
    train_states = np.load(data_dir / "state_train.npy")

    # Generate validation set
    val_obs, val_states = generate_validation_set()

    logger.info(
        f"Valid data shape: {val_obs.shape}, valid states shape: {val_states.shape}"
    )

    # # Use original validation set instead
    # val_obs = np.load(data_dir / "obs_val.npy")
    # val_states = np.load(data_dir / "state_val.npy")

    # Create white noise directory
    noise_dir_name = f"white_noise"
    noise_data_dir = (
        data_dir / noise_dir_name / f"{args.white_noise_dataset_size}-examples"
    )
    noise_data_dir.mkdir(parents=True, exist_ok=True)

    # Sample validation indices once (reuse for all batches)
    # val_batch_indices = random_state.choice(
    #     len(val_states), size=args.white_noise_dataset_size, replace=False
    # )
    val_batch_indices = np.arange(len(val_states))
    common_config = {
        "input_dim": 1,
        "output_dim": 1,
        "input_vocab_size": len(np.unique(train_obs)),
        "block_size": len(train_obs[0]) - 1,
        "num_data_points": args.white_noise_dataset_size,
    }

    # Generate and save white noise batches
    for batch_idx in tqdm.trange(args.num_white_noise_datasets):
        # Sample indices for this batch
        train_batch_indices = random_state.choice(
            len(train_states), size=args.white_noise_dataset_size, replace=False
        )

        # Get batch observations and states
        train_batch_obs = train_obs[train_batch_indices]
        val_batch_obs = val_obs[val_batch_indices]
        train_batch_states = train_states[train_batch_indices]
        val_batch_states = val_states[val_batch_indices]

        # Generate noise for training and validation batches
        train_noise, train_noise_indices, state_to_noise = (
            generate_state_dependent_noise(
                train_batch_states,
                args.seed + batch_idx * 100,
            )
        )
        val_noise, val_noise_indices, _ = generate_state_dependent_noise(
            val_batch_states,
            args.seed + batch_idx * 100 + 50,  # Different seed for val
            state_to_noise,
        )

        # Save noise batches and corresponding data
        np.save(
            noise_data_dir / f"white_noise_output_train_{batch_idx}.npy",
            train_noise,
        )
        np.save(
            noise_data_dir / f"white_noise_output_val_{batch_idx}.npy",
            val_noise,
        )
        np.save(
            noise_data_dir / f"white_noise_obs_train_{batch_idx}.npy",
            train_batch_obs,
        )
        np.save(
            noise_data_dir / f"white_noise_obs_val_{batch_idx}.npy",
            val_batch_obs,
        )
        np.save(
            noise_data_dir / f"white_noise_states_train_{batch_idx}.npy",
            train_batch_states,
        )
        np.save(
            noise_data_dir / f"white_noise_states_val_{batch_idx}.npy",
            val_batch_states,
        )
        np.save(
            noise_data_dir / f"white_noise_indices_train_{batch_idx}.npy",
            train_noise_indices,
        )
        np.save(
            noise_data_dir / f"white_noise_indices_val_{batch_idx}.npy",
            val_noise_indices,
        )

    # Save config
    _, pad_id = get_word_to_id_mapping()
    config = {
        **common_config,
        "output_vocab_size": 2,
        "mask_id": pad_id,
        "predict_type": f"white_noise",
    }

    config_name = f"white_noise_config.yaml"
    with open(config_dir / config_name, "w") as f:
        yaml.dump(config, f)

    logger.info(f"Saved white noise data to {noise_data_dir}")


if __name__ == "__main__":
    main()