import argparse
import logging
import yaml
import torch
import wandb

from inductivebiasprobes.paths import (
    GRIDWORLD_CKPT_DIR,
    GRIDWORLD_CONFIG_DIR,
    GRIDWORLD_DATA_DIR,
    GRIDWORLD_EXT_DIR,
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


def parse_gridworld_args():
    """Parse gridworld-specific command line arguments."""
    parser = argparse.ArgumentParser(description="Train a gridworld model")
    parser = add_common_args(parser)

    # Add gridworld-specific arguments
    parser.add_argument(
        "--num_states",
        type=int,
        help="Number of states in the gridworld",
        default=5,
    )
    return parser.parse_args()


def load_config(args):
    """Load configuration from file and command line args."""
    config = vars(args)
    assert args.config is not None, "Config file is required"
    with (
        GRIDWORLD_CONFIG_DIR / f"{args.num_states}-states" / (args.config + ".yaml")
    ).open("r") as f:
        file_config = yaml.load(f, Loader=yaml.FullLoader)
    file_config.update(config)
    # For Gridworld, set RNN and LSTM to have 2 layers instead of 12
    if file_config["model_type"] in ("rnn", "lstm"):
        file_config["n_layer"] = 2

    return file_config


def train_and_save_model(
    config,
    pretrained_ckpt_dir,
    save_ckpt_dir,
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
    data_dir = GRIDWORLD_DATA_DIR / f"{config['num_states']}-states"
    config["train_file"] = data_dir / "obs_train.npy"
    config["val_file"] = data_dir / "obs_val.npy"
    config["use_float_x"] = False
    config["use_float_y"] = config["output_vocab_size"] is None
    if config["predict_type"] == "state":
        config["train_target_file"] = data_dir / "state_train.npy"
        config["val_target_file"] = data_dir / "state_val.npy"
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
    config["wandb_project"] = f"iclr-gridworld-pretrain-{config['predict_type']}"
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

    def misclassification_callback(output, targets):
        # Compute misclassification rate
        top_preds = torch.argmax(output, dim=-1).view(-1)
        targets = targets.view(-1)
        misclassification_rate = (top_preds != targets).float().mean()
        return misclassification_rate

    loss_callback = misclassification_callback
    loss_callback_name = "miscl_rate"
    if "white_noise" in config["predict_type"]:
        loss_callback = None
        loss_callback_name = None

    # Train model
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
        loss_callback=loss_callback,
        loss_callback_name=loss_callback_name,
    )

    # For white noise models, generate and save extrapolations
    if "white_noise" in config["predict_type"]:
        ext_dir = (
            GRIDWORLD_EXT_DIR
            / f"{config['num_states']}-states"
            / config["predict_type"]
        )
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
    args = parse_gridworld_args()
    config = load_config(args)

    # Setup pretrained checkpoint directory
    curr_ckpt_dir = GRIDWORLD_CKPT_DIR / f"{config['num_states']}-states"
    if config["pretrained"] == "scratch":
        pretrained_ckpt_dir = (
            curr_ckpt_dir / config["model_type"] / config["predict_type"]
        )
    else:
        pretrained_ckpt_dir = (
            curr_ckpt_dir / config["model_type"] / config["pretrained"]
        )

    if "white_noise" in config["predict_type"]:
        for dataset_idx in range(config["num_white_noise_datasets"]):
            logger.info(f"[Training on white noise dataset {dataset_idx}]")

            save_ckpt_dir = (
                curr_ckpt_dir
                / config["model_type"]
                / f"{config['pretrained']}_pt_{config['predict_type']}_idx_{dataset_idx}_transfer"
            )
            run_name = (
                f"{config['white_noise_dataset_size']}_examples_"
                f"{config['max_iters']}_iters_dataset_{dataset_idx}"
            )

            train_and_save_model(
                config,
                pretrained_ckpt_dir,
                save_ckpt_dir,
                run_name=run_name,
                white_noise_dataset_idx=dataset_idx,
            )
    else:
        # Setup checkpoint directories
        if config["pretrained"] == "scratch":
            save_ckpt_dir = (
                curr_ckpt_dir / config["model_type"] / config["predict_type"]
            )
        else:
            save_ckpt_dir = (
                curr_ckpt_dir
                / config["model_type"]
                / f"{config['pretrained']}_pt_{config['predict_type']}_transfer"
            )

        train_and_save_model(
            config,
            pretrained_ckpt_dir,
            save_ckpt_dir,
            run_name=f'pt_{config["pretrained"]}_{config["model_type"]}',
        )


if __name__ == "__main__":
    main()
