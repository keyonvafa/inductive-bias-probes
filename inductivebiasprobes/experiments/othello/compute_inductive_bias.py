import argparse
import json
import logging

import numpy as np
from scipy.spatial.distance import pdist

from inductivebiasprobes.paths import OTHELLO_EXT_DIR, OTHELLO_DATA_DIR

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
        default="scratch",
        choices=["scratch", "next_token"],
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

    return parser.parse_args()


def compute_inductive_bias(ext_test, states, pseudo_states):
    """
    Compute inductive bias.

    Args:
        ext_test: Binary array of shape (n, k') where k' is the number of test seeds
        states: Array of shape (n,) containing the state labels
        pseudo_states: Array of shape (n,) containing the pseudo-state labels

    Returns:
        (same_state_value, diff_state_value)
    """
    n = len(states)

    # Create upper triangular mask to avoid duplicate pairs and self-pairs
    triu_mask = np.triu(np.ones((n, n)), k=1).astype(bool)

    # Create masks for same-state vs. different-state pairs
    state_matrix = states[:, np.newaxis] == states[np.newaxis, :]
    pseudo_state_matrix = pseudo_states[:, np.newaxis] == pseudo_states[np.newaxis, :]
    same_state_mask = state_matrix & triu_mask
    diff_state_mask = (~state_matrix) & triu_mask
    diff_state_same_pseudo_state_mask = diff_state_mask & pseudo_state_matrix
    diff_state_diff_pseudo_state_mask = diff_state_mask & (~pseudo_state_matrix)

    # Print number of same-state and different-state pairs
    logger.info(f"Number of same-state pairs: {np.sum(same_state_mask)}")
    logger.info(f"Number of different-state pairs: {np.sum(diff_state_mask)}")
    logger.info(
        f"Number of different-state pairs with same pseudo-state: "
        f"{np.sum(diff_state_same_pseudo_state_mask)}"
    )
    logger.info(
        f"Number of different-state pairs with different pseudo-state: "
        f"{np.sum(diff_state_diff_pseudo_state_mask)}"
    )

    # Predict exactly x1 for x2 (i.e., 0 -> 0 and 1 -> 1), and compute average accuracy.
    x1_test_expanded = ext_test[:, np.newaxis, :]  # (n, 1, k')
    x2_test_expanded = ext_test[np.newaxis, :, :]  # (1, n, k')

    # Predictions are exactly x1
    # shape: (n, n, k')
    pred_expanded = np.where(x1_test_expanded == 1, 1, 0)

    # Compare predictions to x2 for correctness (boolean)
    losses = (pred_expanded != x2_test_expanded).astype(np.float64)

    # Calculate IB metrics
    same_state_means_per_seed = 2 * (0.5 - losses[same_state_mask]).mean(axis=0)  # (k')
    same_state_value = same_state_means_per_seed.mean()
    same_state_std_dev = same_state_means_per_seed.std(ddof=1)
    same_state_stderr = same_state_std_dev / np.sqrt(same_state_means_per_seed.shape[0])

    diff_state_ib = 2 * losses[diff_state_mask]
    diff_state_means_per_seed = diff_state_ib.mean(axis=0)  # (k')
    diff_state_value = diff_state_means_per_seed.mean()
    diff_state_std_dev = diff_state_means_per_seed.std(ddof=1)
    diff_state_stderr = diff_state_std_dev / np.sqrt(diff_state_means_per_seed.shape[0])

    diff_state_same_pseudo_state_means_per_seed = (
        2 * losses[diff_state_same_pseudo_state_mask]
    ).mean(axis=0)  # (k',)
    diff_state_same_pseudo_state_value = (
        diff_state_same_pseudo_state_means_per_seed.mean()
    )
    diff_state_same_pseudo_state_std_dev = (
        diff_state_same_pseudo_state_means_per_seed.std(ddof=1)
    )
    diff_state_same_pseudo_state_stderr = (
        diff_state_same_pseudo_state_std_dev
        / np.sqrt(diff_state_same_pseudo_state_means_per_seed.shape[0])
    )

    diff_state_diff_pseudo_state_means_per_seed = (
        2 * losses[diff_state_diff_pseudo_state_mask]
    ).mean(axis=0)  # (k',)
    diff_state_diff_pseudo_state_value = (
        diff_state_diff_pseudo_state_means_per_seed.mean()
    )
    diff_state_diff_pseudo_state_std_dev = (
        diff_state_diff_pseudo_state_means_per_seed.std(ddof=1)
    )
    diff_state_diff_pseudo_state_stderr = (
        diff_state_diff_pseudo_state_std_dev
        / np.sqrt(diff_state_diff_pseudo_state_means_per_seed.shape[0])
    )

    return (
        same_state_value,
        same_state_stderr,
        diff_state_value,
        diff_state_stderr,
        diff_state_same_pseudo_state_value,
        diff_state_same_pseudo_state_stderr,
        diff_state_diff_pseudo_state_value,
        diff_state_diff_pseudo_state_stderr,
        diff_state_mask,
        diff_state_ib.mean(axis=1),
    )


def hamming_distance(state1, state2):
    """
    Compute Hamming distance between two board states.

    Args:
        state1, state2: Board states represented as flat arrays

    Returns:
        Integer representing the number of positions that differ
    """
    return np.sum(state1 != state2)


def convert_to_player_perspective(state, player_color=None):
    """
    Convert from standard representation (black=1, white=2, empty=0)
    to player perspective (my color=1, opponent=2, empty=0).

    Args:
        state: Board state represented as a flat array
        player_color: The color of the player (1 for black, 2 for white).
                     If None, automatically determine whose turn it is.

    Returns:
        Converted board state in player perspective
    """
    # Create a copy of the state to avoid modifying the original
    player_perspective_state = np.copy(state)

    # If player_color is not specified, determine whose turn it is
    if player_color is None:
        player_color = determine_player_turn(state)

    # If the player is black (1), the representation is already correct
    if player_color == 1:
        return player_perspective_state

    # If the player is white (2), swap the colors
    player_perspective_state[state == 1] = 2  # Black becomes opponent
    player_perspective_state[state == 2] = 1  # White becomes player

    return player_perspective_state


def build_pseudo_states(state, state_to_id):
    """
    Construct "pseudo-states" from the given board state by
    looking up their next valid moves. Each pseudo-state is
    an equivalence class determined by the set of valid moves.

    Args:
        state: A single board state
        state_to_id: A dictionary mapping board states to integer IDs

    Returns:
        A single integer ID representing the pseudo-state equivalence class.
    """
    from othello_world.data.othello import OthelloBoardState

    board = (state - 1).reshape(8, 8)
    game = OthelloBoardState()
    game.state = np.copy(board)
    valid_moves = frozenset(game.get_valid_moves())
    if valid_moves not in state_to_id:
        state_to_id[valid_moves] = len(state_to_id)
    return state_to_id[valid_moves]


def determine_player_turn(state):
    """
    Determine which player's turn it is based on the board state.

    Args:
        state: Board state represented as a flat array (0 for empty, 1 for black, 2 for white)

    Returns:
        The player color (1 for black, 2 for white) whose turn it is.
    """
    # Count the total number of pieces on the board
    total_pieces = np.sum((state == 1) | (state == 2))

    # In Othello, starting with 4 pieces (2 black, 2 white):
    # - If total pieces is even, it's black's turn (black goes first)
    # - If total pieces is odd, it's white's turn
    return 1 if total_pieces % 2 == 0 else 2


def build_distance_tables(states):
    """
    Build pairwise distance lookup tables for different representations.

    Args:
        states: Array of board states in original representation

    Returns:
        Dictionary containing pairwise distance matrices for different representations
    """
    n = len(states)

    # Initialize distance matrices
    standard_distances = np.zeros((n, n))
    player_perspective_distances = np.zeros((n, n))

    # Compute all pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            # Standard representation distance
            standard_distances[i, j] = standard_distances[j, i] = hamming_distance(
                states[i], states[j]
            )

            # Player perspective representation
            # Convert both states to player perspective based on whose turn it is in each state
            state_i_pp = convert_to_player_perspective(states[i])
            state_j_pp = convert_to_player_perspective(states[j])

            player_perspective_distances[i, j] = player_perspective_distances[j, i] = (
                hamming_distance(state_i_pp, state_j_pp)
            )

    distance_tables = {
        "standard": standard_distances,
        "player_perspective": player_perspective_distances,
    }

    return distance_tables


def main():
    args = parse_args()

    # Set up directory paths
    ext_dir = OTHELLO_EXT_DIR / "synthetic_othello" / "white_noise"
    ext_curr_dir = (
        ext_dir
        / args.model_type
        / f"pt_{args.pretrained}"
        / f"{args.white_noise_dataset_size}_examples"
        / f"{args.max_iters}_iters"
    )

    # Load the original states
    original_states = (
        np.load(ext_dir / "states.npy").astype(np.int32)[:, 5:10, :].reshape(-1, 64)
    )

    # Original discrete IB computation
    random_state = np.random.RandomState(args.seed)
    test_seeds = random_state.choice(
        range(args.num_white_noise_datasets),
        args.num_white_noise_datasets,
        replace=False,
    )
    all_extrapolations = []

    # Build distance tables for different representations
    logger.info("Building distance tables for different representations...")
    distance_tables = build_distance_tables(original_states)
    logger.info("Distance tables built successfully.")

    # Continue with the rest of the processing
    state_to_id = {}
    pseudo_states = np.array(
        [build_pseudo_states(s, state_to_id) for s in original_states]
    )
    state_tuples = [tuple(row) for row in original_states]
    unique_states = {
        state: idx for idx, state in enumerate(sorted(set(state_tuples)))
    }
    states = np.array([unique_states[s] for s in state_tuples])

    for seed in range(args.num_white_noise_datasets):
        # Load data
        ext_probs = np.load(ext_curr_dir / f"idx_{seed}" / "extrapolations.npy")
        extrapolations = np.argmax(ext_probs, axis=-1)[:, 5:10].ravel()
        all_extrapolations.append(extrapolations)

    # Stack arrays from different seeds
    all_extrapolations = np.stack(
        all_extrapolations, axis=1
    )  # Shape: (num_examples, num_seeds)
    ext_test = all_extrapolations[:, test_seeds]

    # Compute inductive bias
    if len(states) <= args.num_examples:
        logger.info(f"Using all {len(states)} examples")
        example_indices = np.arange(len(states))
    else:
        logger.info(f"Using {args.num_examples} random examples")
        example_indices = random_state.choice(
            range(len(states)), size=args.num_examples, replace=False
        )

    # Subset all relevant arrays
    ext_test = ext_test[example_indices]
    states = states[example_indices]
    pseudo_states = pseudo_states[example_indices]

    # Subset distance tables
    subset_distance_tables = {}
    for rep, dist_matrix in distance_tables.items():
        subset_distance_tables[rep] = dist_matrix[
            np.ix_(example_indices, example_indices)
        ]

    # Compute inductive bias
    (
        same_state_value,
        same_state_stderr,
        diff_state_value,
        diff_state_stderr,
        diff_state_same_pseudo_state_value,
        diff_state_same_pseudo_state_stderr,
        diff_state_diff_pseudo_state_value,
        diff_state_diff_pseudo_state_stderr,
        diff_state_mask,
        diff_state_ib,
    ) = compute_inductive_bias(
        ext_test, states, pseudo_states
    )
    logger.info(f"Inductive Bias Results:")
    logger.info(
        f"Average IB for same state pairs: {same_state_value:.3f} "
        f"± {same_state_stderr:.3f}"
    )
    logger.info(
        f"Average IB loss for different state pairs with same pseudo-state: "
        f"{diff_state_same_pseudo_state_value:.3f} ± {diff_state_same_pseudo_state_stderr:.3f}"
    )
    logger.info(
        f"Average IB loss for different state pairs with different pseudo-state: "
        f"{diff_state_diff_pseudo_state_value:.3f} ± {diff_state_diff_pseudo_state_stderr:.3f}"
    )
    logger.info(
        f"Average IB loss for different state pairs: {diff_state_value:.3f} "
        f"± {diff_state_stderr:.3f}"
    )
    logger.info("-" * 80)

    # Save results
    ib_results = {
        "same_state_ib": same_state_value,
        "same_state_stderr": same_state_stderr,
        "diff_state_loss": diff_state_value,
        "diff_state_stderr": diff_state_stderr,
        "diff_state_same_pseudo_state_loss": diff_state_same_pseudo_state_value,
        "diff_state_same_pseudo_state_stderr": diff_state_same_pseudo_state_stderr,
        "diff_state_diff_pseudo_state_loss": diff_state_diff_pseudo_state_value,
        "diff_state_diff_pseudo_state_stderr": diff_state_diff_pseudo_state_stderr,
    }
    with open(
        ext_curr_dir / f"ib.json", "w"
    ) as f:
        json.dump(ib_results, f, indent=4)

    # Compute correlations between distances and inductive bias metrics
    compute_and_save_correlations(
        diff_state_mask, diff_state_ib, subset_distance_tables, ext_curr_dir
    )


def compute_and_save_correlations(mask, ib, distance_tables, output_dir):
    """
    Compute correlations between distances and inductive bias metrics.

    Args:
        mask: Boolean mask of shape (n, n)
        ib: Array of shape (n, k)
        distance_tables: Dictionary containing distance matrices for different representations
        output_dir: Directory to save the correlation results
    """
    import scipy.stats as stats

    logger.info("Computing correlations between distances and inductive bias metrics")
    correlation_data = {}

    for rep_name, rep_data in distance_tables.items():
        distances = rep_data[mask]
        corr, _ = stats.pearsonr(distances, ib)
        correlation_data[rep_name] = corr

    # Save correlation results
    with open(output_dir / "distance_ib_correlations.json", "w") as f:
        json.dump(correlation_data, f, indent=4)

    # Log summary of correlations
    logger.info("Correlation Summary:")
    for rep_name, rep_correlations in correlation_data.items():
        logger.info(f"Representation: {rep_name}")
        logger.info(f"  Correlation: {rep_correlations}")


if __name__ == "__main__":
    main()
