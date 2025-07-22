import argparse
import json
import logging

import numpy as np

from inductivebiasprobes.paths import GRIDWORLD_EXT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_states",
        type=int,
        help="Number of states in the gridworld",
        default=5,
    )
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

    return parser.parse_args()


def build_pseudo_states(state, state_to_id):
    """
    Construct "pseudo-states" from the given board state by
    looking up their next valid moves. Each pseudo-state is
    an equivalence class determined by the set of valid moves.
    """
    if state == 0:
        valid_moves = frozenset([1, 2])
    elif state == 4:
        valid_moves = frozenset([0, 1])
    else:
        valid_moves = frozenset([0, 1, 2])
    if valid_moves not in state_to_id:
        state_to_id[valid_moves] = len(state_to_id)
    return state_to_id[valid_moves]


def _compute_discrete_inductive_bias(
    ext_train, ext_test, states, pseudo_states
):
    """
    Helper function to compute discrete inductive bias.

    Args:
        ext_train: Binary array of shape (n, k) where n is number of points and k is number of train seeds
        ext_test: Binary array of shape (n, k') where k' is number of test seeds
        states: Array of shape (n,) containing the state labels
        pseudo_states: Array of shape (n,) containing the pseudo-state labels

    Returns:
        (same_state_value, diff_state_value)
    """
    n = len(states)

    # Create upper triangular mask to avoid duplicate pairs and self-pairs
    triu_mask = np.triu(np.ones((n, n)), k=1).astype(bool)

    # Create state comparison matrix
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
    # In "strong", predict exactly x1 for x2 (i.e., 0 -> 0 and 1 -> 1),
    # and then compute average accuracy.
    x1_test_expanded = ext_test[:, np.newaxis, :]  # (n, 1, k')
    x2_test_expanded = ext_test[np.newaxis, :, :]  # (1, n, k')

    # Predictions are exactly x1
    # shape: (n, n, k')
    pred_expanded = np.where(x1_test_expanded == 1, 1, 0)

    # Compare predictions to x2 for correctness (boolean)
    losses = (pred_expanded != x2_test_expanded).astype(np.float64)

    same_state_means_per_seed = (2 * (0.5 - losses[same_state_mask])).mean(
        axis=0
    )  # (k')
    same_state_value = same_state_means_per_seed.mean()
    same_state_std_dev = same_state_means_per_seed.std(ddof=1)
    same_state_stderr = same_state_std_dev / np.sqrt(same_state_means_per_seed.shape[0])

    diff_state_means_per_seed = (2 * losses[diff_state_mask]).mean(axis=0)  # (k')
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
    )


def compute_discrete_inductive_bias(args, states, ext_curr_dir):
    """
    Compute and save discrete inductive bias.

    Args:
        args: Arguments
    """
    test_frac = 1.0
    random_state = np.random.RandomState(args.seed)
    test_seeds = random_state.choice(
        range(args.num_white_noise_datasets),
        int(test_frac * args.num_white_noise_datasets),
        replace=False,
    )
    train_seeds = np.setdiff1d(range(args.num_white_noise_datasets), test_seeds)
    all_extrapolations = []
    state_to_id = {}
    pseudo_states = np.array([build_pseudo_states(s, state_to_id) for s in states])

    for seed in range(args.num_white_noise_datasets):
        # Load data
        ext_probs = np.load(ext_curr_dir / f"idx_{seed}" / "extrapolations.npy")
        extrapolations = np.argmax(ext_probs, axis=-1)
        flattened_extrapolations = extrapolations.ravel()
        all_extrapolations.append(flattened_extrapolations)

    # Stack arrays from different seeds
    all_extrapolations = np.stack(
        all_extrapolations, axis=1
    )  # Shape: (num_examples, num_seeds)
    ext_train = all_extrapolations[:, train_seeds]
    ext_test = all_extrapolations[:, test_seeds]

    # Compute inductive bias
    example_indices = random_state.choice(
        range(len(states)), size=args.num_examples, replace=False
    )
    ext_train, ext_test, states, pseudo_states = (
        ext_train[example_indices],
        ext_test[example_indices],
        states[example_indices],
        pseudo_states[example_indices],
    )
    (
        same_state_value,
        same_state_stderr,
        diff_state_value,
        diff_state_stderr,
        diff_state_same_pseudo_state_value,
        diff_state_same_pseudo_state_stderr,
        diff_state_diff_pseudo_state_value,
        diff_state_diff_pseudo_state_stderr,
    ) = _compute_discrete_inductive_bias(
        ext_train, ext_test, states, pseudo_states
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
    with open(ext_curr_dir / f"ib.json", "w") as f:
        json.dump(ib_results, f, indent=4)


def main():
    args = parse_args()
    ext_dir = (
        GRIDWORLD_EXT_DIR
        / f"{args.num_states}-states"
        / f"white_noise"
    )
    states = np.load(ext_dir / "states.npy").astype(np.int32)
    ext_curr_dir = (
        ext_dir
        / args.model_type
        / f"pt_{args.pretrained}"
        / f"{args.white_noise_dataset_size}_examples"
        / f"{args.max_iters}_iters"
    )
    compute_discrete_inductive_bias(
        args, states.ravel(), ext_curr_dir
    )


if __name__ == "__main__":
    main()
