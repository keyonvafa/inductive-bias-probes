import argparse
import logging
import yaml
import torch
import numpy as np
import wandb

from othello_world.data.othello import OthelloBoardState
from inductivebiasprobes.paths import (
    OTHELLO_CKPT_DIR,
    OTHELLO_CONFIG_DIR,
    OTHELLO_DATA_DIR,
    OTHELLO_EXT_DIR,
)
from inductivebiasprobes.src.train_utils import (
    add_common_args,
    generate_and_save_extrapolations,
    init_model,
    setup_training_environment,
    train,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_othello_args():
    """Parse othello-specific command line arguments."""
    parser = argparse.ArgumentParser(description="Train an othello model")
    parser = add_common_args(parser)

    # Add othello-specific arguments
    parser.add_argument(
        "--reconstruct_board",
        action="store_true",
        help="Whether to evaluate and save board reconstruction metrics",
    )
    parser.add_argument(
        "--reconstruction_eval_interval",
        type=int,
        default=1000,
        help="Interval (in iterations) at which to evaluate board reconstruction",
    )
    return parser.parse_args()


def load_config(args):
    """Load configuration from file and command line args."""
    config = vars(args)
    assert args.config is not None, "Config file is required"
    config_path = (
        OTHELLO_CONFIG_DIR / "synthetic_othello" / f"{args.config}.yaml"
    )
    with config_path.open("r") as f:
        file_config = yaml.load(f, Loader=yaml.FullLoader)
    file_config.update(config)
    return file_config


def evaluate_board_reconstruction(model, val_loader, config, iter_num):
    """Evaluate board reconstruction metrics."""
    # Store original mode
    was_training = model.training
    model.eval()
    all_preds = []
    all_states = []

    with torch.no_grad():
        for batch in val_loader:
            x, states = batch
            x = x.to(config["device"])
            states = states.to(config["device"])

            # Get model predictions
            state_preds = model(x)
            state_preds = state_preds.view(-1, model.config.output_vocab_size)
            top_preds = torch.argmax(state_preds, dim=-1)
            top_preds = top_preds.view(states.shape)

            all_preds.append(top_preds.cpu().numpy())
            all_states.append(states.cpu().numpy())

    # Concatenate and combine first two dimensions
    all_preds = np.concatenate(all_preds, axis=0)
    all_preds = all_preds.reshape(-1, all_preds.shape[-1])
    all_states = np.concatenate(all_states, axis=0)
    all_states = all_states.reshape(-1, all_states.shape[-1])

    # Calculate metrics
    tile_accuracy = (all_states == all_preds).mean()
    board_accuracy = ((all_states == all_preds).mean(-1) == 1).mean()

    # Calculate valid moves metrics
    total = len(all_states)
    total_same_next_moves = 0
    total_same_boards = 0
    total_one_move_in_common = 0
    total_reconstructed_is_subset = 0

    for pred_board, true_board in zip(all_preds, all_states):
        pred_board = (pred_board - 1).reshape(8, 8)
        true_board = (true_board - 1).reshape(8, 8)

        # Initialize board states
        pred_game = OthelloBoardState()
        true_game = OthelloBoardState()
        pred_game.state = np.copy(pred_board)
        true_game.state = np.copy(true_board)

        # Set next hand color based on board state
        next_hand_color = 1 if np.sum(np.abs(true_board) == 1) % 2 == 0 else -1
        pred_game.next_hand_color = next_hand_color
        true_game.next_hand_color = next_hand_color

        # Get valid moves
        valid_moves_pred = set(pred_game.get_valid_moves())
        valid_moves_true = set(true_game.get_valid_moves())

        # Calculate metrics
        if valid_moves_pred == valid_moves_true:
            total_same_next_moves += 1
        if np.all(pred_board == true_board):
            total_same_boards += 1
        if len(valid_moves_pred & valid_moves_true) > 0:
            total_one_move_in_common += 1
        if valid_moves_pred | valid_moves_true == valid_moves_true:
            total_reconstructed_is_subset += 1

    metrics = {
        "transfer_steps": iter_num,
        "tile_accuracy": tile_accuracy,
        "board_accuracy": board_accuracy,
        "same_board_frac": total_same_boards / total,
        "next_move_frac": total_same_next_moves / total,
        "one_move_in_common_frac": total_one_move_in_common / total,
        "reconstructed_is_subset_frac": total_reconstructed_is_subset / total,
    }

    # Log to wandb if enabled
    if not config["no_wandb"]:
        wandb.log(
            {
                "reconstruction/same_board_frac": metrics["same_board_frac"],
                "reconstruction/next_move_frac": metrics["next_move_frac"],
                "reconstruction/one_move_in_common_frac": metrics[
                    "one_move_in_common_frac"
                ],
                "reconstruction/reconstructed_is_subset_frac": metrics[
                    "reconstructed_is_subset_frac"
                ],
            }
        )

    # Restore original mode
    if was_training:
        model.train()

    return metrics


def train_and_save_model(
    config,
    pretrained_ckpt_dir,
    save_ckpt_dir,
    save_callback_dir,
    run_name=None,
    white_noise_dataset_idx=None,
):
    """Initialize, train and optionally save model predictions."""
    save_checkpoints = "white_noise" not in config["predict_type"]

    # Setup training environment
    ddp, master_process, ptdtype, config = setup_training_environment(
        config, save_ckpt_dir, save_checkpoints
    )

    # Create dataloaders
    data_dir = OTHELLO_DATA_DIR / "synthetic_othello"
    config["train_file"] = data_dir / "obs_train.npy"
    config["val_file"] = data_dir / "obs_val.npy"
    config["use_float_x"] = False
    config["use_float_y"] = config["output_vocab_size"] is None
    if config["predict_type"] == "state":
        config["train_target_file"] = data_dir / "state_train.npy"
        config["val_target_file"] = data_dir / "state_val.npy"
    elif config["predict_type"] in ("balance-black", "edges", "majority", "parity"):
        curr_data_dir = (
            data_dir / f"transfer_{config['predict_type'].replace('-', '_')}"
        )
        config["train_file"] = curr_data_dir / "obs_train.npy"
        config["val_file"] = curr_data_dir / "obs_val.npy"
        config["train_target_file"] = curr_data_dir / "state_train.npy"
        config["val_target_file"] = curr_data_dir / "state_val.npy"
    elif "white_noise" in config["predict_type"]:
        config["train_file"] = (
            data_dir
            / config["predict_type"]
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_obs_train_{white_noise_dataset_idx}.npy"
        )
        config["val_file"] = (
            data_dir
            / config["predict_type"]
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_obs_val_{white_noise_dataset_idx}.npy"
        )
        config["train_target_file"] = (
            data_dir
            / config["predict_type"]
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_output_train_{white_noise_dataset_idx}.npy"
        )
        config["val_target_file"] = (
            data_dir
            / config["predict_type"]
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_states_val_{white_noise_dataset_idx}.npy"
        )
        config["train_indices_file"] = (
            data_dir
            / config["predict_type"]
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_indices_train_{white_noise_dataset_idx}.npy"
        )
        config["val_indices_file"] = (
            data_dir
            / config["predict_type"]
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_indices_val_{white_noise_dataset_idx}.npy"
        )

    # Setup wandb config
    config["wandb_project"] = (
        f"iclr-synthetic-othello-pretrain-{config['predict_type']}"
    )
    config["wandb_entity"] = "petergchang"
    config["wandb_run_name"] = run_name or "gpt"
    if "white_noise" in config["predict_type"]:
        config["no_wandb"] = True

    # Initialize model
    model, config, iter_num, current_epoch, best_val_loss, optimizer, scaler = (
        init_model(
            config=config,
            ckpt_dir=pretrained_ckpt_dir,
            ddp=ddp,
        )
    )

    # Setup wandb logging
    if not config["no_wandb"] and master_process:
        wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            name=config["wandb_run_name"],
            resume="allow",
            config=config,
        )

    # Train model with board reconstruction evaluation if requested
    callback_fn = evaluate_board_reconstruction if config["reconstruct_board"] else None
    callback_interval = (
        config["reconstruction_eval_interval"] if config["reconstruct_board"] else None
    )

    def misclassification_callback(
        output, targets, output_vocab_size=config["output_vocab_size"]
    ):
        # Compute misclassification rate
        if output.shape[-1] != output_vocab_size:
            output = output.view(*output.shape[:-1], -1, output_vocab_size)
        top_preds = torch.argmax(output, dim=-1).view(targets.shape)
        misclassification_rate = (top_preds != targets).float().mean(axis=(1, 2))
        return misclassification_rate

    train(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        ddp=ddp,
        master_process=master_process,
        ptdtype=ptdtype,
        iter_num=iter_num,
        current_epoch=current_epoch,
        best_val_loss=best_val_loss,
        ckpt_dir=save_ckpt_dir,
        save_checkpoints=save_checkpoints,
        callback_fn=callback_fn,
        callback_interval=callback_interval,
        callback_dir=save_callback_dir,
        loss_callback=misclassification_callback,
        loss_callback_name="miscl_rate",
    )

    # For white noise models, generate and save extrapolations
    if "white_noise" in config["predict_type"]:
        ext_dir = OTHELLO_EXT_DIR / "synthetic_othello" / config["predict_type"]
        ext_idx_dir = (
            ext_dir
            / config["model_type"]
            / f"pt_{config['pretrained']}"
            / f"{config['white_noise_dataset_size']}_examples"
            / f"{config['max_iters']}_iters"
            / f"idx_{white_noise_dataset_idx}"
        )

        generate_and_save_extrapolations(model, config, ext_dir, ext_idx_dir)

    if not config["no_wandb"] and master_process:
        wandb.finish()

    return model


def main():
    # Parse arguments and load config
    args = parse_othello_args()
    config = load_config(args)

    # Setup pretrined checkpoint directory
    save_callback_dir = None
    if config["pretrained"] == "scratch":
        pretrained_ckpt_dir = (
            OTHELLO_CKPT_DIR / "synthetic_othello" / config["model_type"] / config["predict_type"]
        )
    else:
        pretrained_ckpt_dir = (
            OTHELLO_CKPT_DIR
            / "synthetic_othello"
            / config["model_type"]
            / config["pretrained"]
        )

    if "white_noise" in config["predict_type"]:
        for dataset_idx in range(config["num_white_noise_datasets"]):
            logger.info(f"Training on white noise dataset {dataset_idx}")

            # Build checkpoint path
            ckpt_name = f"{config['pretrained']}_pt_{config['predict_type']}"
            ckpt_name += f"_idx_{dataset_idx}_transfer"

            save_ckpt_dir = (
                OTHELLO_CKPT_DIR
                / "synthetic_othello"
                / config["model_type"]
                / ckpt_name
            )

            # Build run name
            run_name = (
                f"{config['white_noise_dataset_size']}_examples_"
                f"{config['max_iters']}_iters_dataset_{dataset_idx}"
            )

            train_and_save_model(
                config,
                pretrained_ckpt_dir,
                save_ckpt_dir,
                save_callback_dir,
                run_name=run_name,
                white_noise_dataset_idx=dataset_idx,
            )
    else:
        # Setup checkpoint directories
        if config["pretrained"] == "scratch":
            save_ckpt_dir = (
                OTHELLO_CKPT_DIR
                / "synthetic_othello"
                / config["model_type"]
                / config["predict_type"]
            )
        else:
            save_ckpt_dir = (
                OTHELLO_CKPT_DIR
                / "synthetic_othello"
                / config["model_type"]
                / f"{config['pretrained']}_pt_{config['predict_type']}_transfer"
            )
            if config["reconstruct_board"] and config["predict_type"] == "state":
                save_callback_dir = save_ckpt_dir

        train_and_save_model(
            config,
            pretrained_ckpt_dir,
            save_ckpt_dir,
            save_callback_dir,
            run_name=f'pt_{config["pretrained"]}_{config["model_type"]}',
        )


if __name__ == "__main__":
    main()
