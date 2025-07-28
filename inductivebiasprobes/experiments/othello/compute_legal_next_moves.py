import argparse
import logging
import torch
import numpy as np
import yaml
from tqdm import tqdm
from othello_world.data.othello import OthelloBoardState

from inductivebiasprobes import Model, ModelConfig
from inductivebiasprobes.paths import (
    OTHELLO_CKPT_DIR,
    OTHELLO_DATA_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_word_to_id_mapping():
    """Create mapping from board positions to token IDs."""
    # The 4 center squares are not valid first moves, so they are not in vocab
    word_numbers = [str(i) for i in range(64) if i not in (27, 28, 35, 36)]
    word_to_id = {word: idx for idx, word in enumerate(word_numbers)}
    pad_id = max(word_to_id.values()) + 1
    word_to_id["<pad>"] = pad_id
    return word_to_id, pad_id


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute legal next moves for Othello models."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["rnn", "lstm", "gpt", "mamba", "mamba2"],
        help="List of model types to evaluate.",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of games to evaluate.",
    )
    return parser.parse_args()


def load_model_and_config(model_type):
    """Load a pretrained model and its configuration."""
    ckpt_dir = OTHELLO_CKPT_DIR / "synthetic_othello" / model_type / "next_token"
    ckpt_path = ckpt_dir / "ckpt.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # This filtering logic is from fit_symbolic_regression.py
    # to ensure only valid args are passed to ModelConfig
    fixed_configs = {
        "model_type", "n_embd", "n_layer", "bias", "input_dim", "block_size",
        "input_vocab_size", "n_head", "dropout", "dt_rank", "d_state",
        "expand_factor", "d_conv", "dt_min", "dt_max", "dt_init", "dt_scale",
        "dt_init_floor", "rms_norm_eps", "conv_bias", "inner_layernorms",
    }
    mutable_configs = {
        "output_dim", "mask_id", "output_vocab_size", "pscan", "use_cuda",
    }
    all_configs = fixed_configs | mutable_configs
    load_model_args = {}
    for k in all_configs:
        if k in checkpoint_model_args:
            load_model_args[k] = checkpoint_model_args[k]

    model_config = ModelConfig(**load_model_args)
    model = Model(model_config)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    config = load_model_args
    config['device'] = device
    return model, config

def main():
    args = parse_args()

    results = {}
    word_to_id, pad_id = get_word_to_id_mapping()
    id_to_word = {v: int(k) for k, v in word_to_id.items() if k != '<pad>'}

    data_dir = OTHELLO_DATA_DIR / "synthetic_othello"
    val_data = np.load(data_dir / "obs_val.npy")
    if val_data.ndim > 2:
        val_data = val_data.squeeze(-1)
    
    games = val_data[:args.num_games]

    print(f"| {'Model':<12} | {'Legal Move Accuracy':<20} |")
    print(f"|{'-'*14}|{'-'*22}|")

    for model_type in args.models:
        model, config = load_model_and_config(model_type)
        
        total_predictions = 0
        correct_predictions = 0

        for game_moves in tqdm(games, desc=f"Evaluating {model_type}", leave=False):
            pad_indices = np.where(game_moves == pad_id)[0]
            game_len = pad_indices[0] if len(pad_indices) > 0 else len(game_moves)

            # we can't make a prediction from the very first token (which is a move)
            for t in range(1, game_len):
                context_token_ids = game_moves[:t]
                
                context_tensor = torch.from_numpy(context_token_ids).long().to(config['device']).unsqueeze(0)
                
                with torch.no_grad():
                    # Add a dimension for the input feature
                    logits = model(context_tensor.unsqueeze(-1))
                    pred_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

                board = OthelloBoardState()
                
                # convert token ids to 0-63 moves and update board
                actual_moves = [id_to_word[m] for m in context_token_ids if m in id_to_word]
                board.update(actual_moves)
                
                valid_moves = board.get_valid_moves()
                
                pred_move = id_to_word.get(pred_token_id)
                
                if pred_move is not None and pred_move in valid_moves:
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        results[model_type] = accuracy
        print(f"| {model_type:<12} | {accuracy:<20.4f} |")

if __name__ == "__main__":
    main() 