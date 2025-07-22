import argparse
import logging
import yaml
import zipfile

import numpy as np
from tqdm import tqdm
from othello_world.data import get_othello
from othello_world.mingpt.dataset import CharDataset
from othello_world.data.othello import OthelloBoardState

from inductivebiasprobes.paths import OTHELLO_CONFIG_DIR, OTHELLO_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def get_word_to_id_mapping():
    """Create mapping from board positions to token IDs."""
    word_numbers = [str(i) for i in range(64) if i not in (27, 28, 35, 36)]
    word_to_id = {word: idx for idx, word in enumerate(word_numbers)}
    pad_id = max(word_to_id.values()) + 1
    word_to_id["<pad>"] = pad_id
    return word_to_id, pad_id


def game_to_states(game):
    """Convert game moves to board states."""
    board = OthelloBoardState()
    states = np.array(board.get_gt(game, "get_state")).astype(np.int8)
    max_moves = 60
    padded_moves = np.pad(
        states,
        ((0, max_moves - len(states)), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return padded_moves


def process_championship_data(
    dataset, dataset_val, word_to_id, pad_id, seq_len, save_states=True,
):
    """Process championship or synthetic Othello data."""
    train_indices = np.arange(len(dataset.data))
    valid_indices = np.arange(len(dataset_val.data))

    train_data = pad_id * np.ones((len(train_indices), seq_len), dtype=np.int8)
    valid_data = pad_id * np.ones((len(valid_indices), seq_len), dtype=np.int8)

    # Tokenize sequences
    for i, index in tqdm(
        enumerate(train_indices),
        total=len(train_indices),
        desc="Processing training data",
    ):
        game = dataset.data[index]
        for j, word in enumerate(game):
            train_data[i, j] = word_to_id.get(str(word), pad_id)

    for i, index in tqdm(
        enumerate(valid_indices),
        total=len(valid_indices),
        desc="Processing validation data",
    ):
        game = dataset_val.data[index]
        for j, word in enumerate(game):
            valid_data[i, j] = word_to_id.get(str(word), pad_id)

    train_states = []
    valid_states = []
    if save_states:
        for index in tqdm(train_indices, desc="Processing training states"):
            game_states = game_to_states(dataset.data[index])
            train_states.append(game_states)

        for index in tqdm(valid_indices, desc="Processing validation states"):
            game_states = game_to_states(dataset_val.data[index])
            valid_states.append(game_states)

    return (
        np.atleast_3d(train_data),
        np.atleast_3d(valid_data),
        train_states,
        valid_states,
    )


def save_data_files(
    data_dir,
    config_dir,
    train_data,
    valid_data,
    train_states,
    valid_states,
    pad_id,
    save_states=True,
):
    """Save processed data and config files."""
    # Save original data
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save original observations
    train_file_name = data_dir / "obs_train.npy"
    np.save(train_file_name, train_data)

    valid_file_name = data_dir / "obs_val.npy"
    np.save(valid_file_name, valid_data)

    # Save configs only for original data
    token_vocab_size = len(np.unique(train_data))
    _, seq_len, _ = train_data.shape
    common_config = {
        "input_dim": 1,
        "input_vocab_size": token_vocab_size,
        "mask_id": pad_id,
        "block_size": seq_len - 1,
        "num_data_points": len(train_data),
    }
    ntp_config = {
        **common_config,
        "output_dim": 1,
        "output_vocab_size": token_vocab_size,
        "predict_type": "next_token",
    }
    state_config = {
        **common_config,
        "output_dim": 64,
        "output_vocab_size": 3,
        "predict_type": "state",
    }

    with open(config_dir / "ntp_config.yaml", "w") as f:
        yaml.dump(ntp_config, f)
    with open(config_dir / "state_config.yaml", "w") as f:
        yaml.dump(state_config, f)

    if save_states:
        all_train_states = np.array(train_states)
        all_valid_states = np.array(valid_states)

        # Save original states
        train_states_file_name = data_dir / "state_train.npy"
        np.save(train_states_file_name, all_train_states)

        valid_states_file_name = data_dir / "state_val.npy"
        np.save(valid_states_file_name, all_valid_states)

        # Generate and save all transformations
        transformations = {
            "parity": parity_transformation,
            "majority": majority_transformation,
            "balance-black": balance_black_transformation,
            "edges": edges_transformation,
        }

        for transform_name, transform_func in transformations.items():
            transfer_name = f"transfer_{transform_name.replace('-', '_')}"
            transform_dir = data_dir / transfer_name
            transform_dir.mkdir(parents=True, exist_ok=True)

            # Save observations (same as original)
            np.save(transform_dir / "obs_train.npy", train_data)
            np.save(transform_dir / "obs_val.npy", valid_data)

            # Transform and save states
            transformed_train_states = transform_func(all_train_states)
            transformed_valid_states = transform_func(all_valid_states)

            np.save(transform_dir / "state_train.npy", transformed_train_states)
            np.save(transform_dir / "state_val.npy", transformed_valid_states)

            logger.info(f"Saved {transform_name} transformed data to {transform_dir}")

            # Save config
            curr_config = {
                **common_config,
                "output_dim": 1,
                "output_vocab_size": len(np.unique(transformed_train_states)),
                "predict_type": transform_name,
            }
            with open(config_dir / f"{transfer_name}_config.yaml", "w") as f:
                yaml.dump(curr_config, f)

    logger.info(f"Saved original data to {data_dir}")


def parity_transformation(boards):
    num_black_pieces = (boards == 0).sum(-1)
    transformed_boards = (num_black_pieces % 2).astype(np.int8)
    return np.atleast_3d(transformed_boards)


def majority_transformation(boards):
    transformed_boards = ((boards.astype(np.int8) - 1).sum(-1) > 0).astype(np.int8)
    return np.atleast_3d(transformed_boards)


def balance_black_transformation(boards):
    top_board_pieces = (boards[:, :, :32] == 0).sum(-1)
    bottom_board_pieces = (boards[:, :, 32:] == 0).sum(-1)
    transformed_boards = (top_board_pieces > bottom_board_pieces).astype(np.int8)
    return np.atleast_3d(transformed_boards)


def edges_transformation(boards):
    """
    Returns 1 if black has more edge squares than white in a given state, else 0.
    The edges are the top row (indices 0..7), bottom row (56..63), left column (multiples of 8),
    and right column (7, 15, 23, ..., 63). This includes the four corners.
    boards shape: (batch, time, 64), with 0=black, 1=white, 2=empty.
    """
    # Compute all edge indices, now including corners
    top_edge = list(range(0, 8))  # 0..7
    bottom_edge = list(range(56, 64))  # 56..63
    left_edge = [i for i in range(8, 56, 8)]  # 8,16,24,32,40,48
    right_edge = [i for i in range(15, 63, 8)]  # 15,23,31,39,47,55
    edge_indices = top_edge + bottom_edge + left_edge + right_edge

    edges_black = (boards[..., edge_indices] == 0).sum(axis=-1)
    edges_white = (boards[..., edge_indices] == 1).sum(axis=-1)
    transformed_boards = (edges_black > edges_white).astype(np.int8)

    return np.atleast_3d(transformed_boards)


def main():
    args = parse_args()

    word_to_id, pad_id = get_word_to_id_mapping()
    seq_len = 60

    data_name = "synthetic-othello".replace("-", "_")
    data_dir = OTHELLO_DATA_DIR / data_name
    config_dir = OTHELLO_CONFIG_DIR / data_name
    data_root = OTHELLO_DATA_DIR / "othello_synthetic"
    if not data_root.exists():
        zip_path_str = "othello_synthetic-20241111T015400Z-"
        zip_path1 = OTHELLO_DATA_DIR / (zip_path_str + "001.zip")
        zip_path2 = OTHELLO_DATA_DIR / (zip_path_str + "002.zip")
        with zipfile.ZipFile(zip_path1, "r") as zip_ref:
            zip_ref.extractall(OTHELLO_DATA_DIR)
        with zipfile.ZipFile(zip_path2, "r") as zip_ref:
            zip_ref.extractall(OTHELLO_DATA_DIR)
    othello_synth = get_othello(ood_num=-1, data_root=data_root, wthor=True)
    othello_synth_valid = othello_synth.val
    dataset_synth = CharDataset(othello_synth)
    dataset_synth_valid = CharDataset(othello_synth_valid)

    train_data, valid_data, train_states, valid_states = process_championship_data(
        dataset_synth, dataset_synth_valid, word_to_id, pad_id, seq_len, True
    )

    save_data_files(
        data_dir,
        config_dir,
        train_data,
        valid_data,
        train_states,
        valid_states,
        pad_id,
        True,
    )


if __name__ == "__main__":
    main()
