import argparse
import logging
import yaml
import wandb

from inductivebiasprobes.paths import (
    PHYSICS_CONFIG_DIR,
    PHYSICS_CKPT_DIR,
    PHYSICS_DATA_DIR,
    PHYSICS_EXT_DIR,
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


def parse_physics_args():
    """Parse physics-specific command line arguments."""
    parser = argparse.ArgumentParser(description="Train a physics model")
    parser = add_common_args(parser)
    parser.add_argument("--white_noise_dataset_idx_lower", type=int, default=None)
    parser.add_argument("--white_noise_dataset_idx_upper", type=int, default=None)
    return parser.parse_args()


def load_config(args):
    """Load configuration from file and command line args."""
    config = vars(args)
    assert args.config is not None, "Config file is required"
    with (PHYSICS_CONFIG_DIR / (args.config + ".yaml")).open("r") as f:
        file_config = yaml.load(f, Loader=yaml.FullLoader)
    file_config.update(config)
    return file_config



def train_and_save_model(
    config,
    pretrained_ckpt_dir,
    save_ckpt_dir,
    run_name=None,
    white_noise_dataset_idx=None,
):
    """Initialize, train and optionally save model predictions."""
    save_checkpoints = config["predict_type"] != "white_noise"
    # Setup training environment
    ddp, master_process, ptdtype, config = setup_training_environment(
        config, save_ckpt_dir, save_checkpoints
    )

    # Create dataloaders
    config["train_file"] = PHYSICS_DATA_DIR / "obs_train.npy"
    config["val_file"] = PHYSICS_DATA_DIR / "obs_val.npy"
    config["test_file"] = PHYSICS_DATA_DIR / "obs_test.npy"
    config["use_float_x"] = False 
    config["use_float_y"] = config["output_vocab_size"] is None
    train_label, val_label = "train", "val"
    if config["predict_type"] == "state":
        config["train_target_file"] = (
            PHYSICS_DATA_DIR / f"{config['predict_type']}_train.npy"
        )
        config["val_target_file"] = (
            PHYSICS_DATA_DIR / f"{config['predict_type']}_val.npy"
        )
    elif "force" in config["predict_type"]:
        if config["predict_type"] == "force_magnitude":
            print("NOTE: USING MASKED DATA")
            config["train_file"] = PHYSICS_DATA_DIR / f"obs_two_body_train.npy"
            config["val_file"] = PHYSICS_DATA_DIR / f"obs_two_body_train.npy"
            config["train_target_file"] = (
                PHYSICS_DATA_DIR / f"force_magnitude_two_body_train_masked.npy"
            )
            config["val_target_file"] = (
                PHYSICS_DATA_DIR / f"force_magnitude_two_body_train.npy"
            )
        elif config["predict_type"] == "force_vector":
            print("NOTE: USING MASKED SOLAR SYSTEM DATA")
            config['train_file'] = PHYSICS_DATA_DIR / f"obs_solar_system_two_body.npy"
            config['val_file'] = PHYSICS_DATA_DIR / f"obs_solar_system_two_body.npy"
            config["train_target_file"] = (
                PHYSICS_DATA_DIR / f"force_vector_solar_system_two_body_masked.npy"
            )
            config["val_target_file"] = (
                PHYSICS_DATA_DIR / f"force_vector_solar_system_two_body.npy"
            )
    elif config["predict_type"] == "white_noise":
        config["train_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_obs_train_{white_noise_dataset_idx}.npy"
        )
        config["val_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_obs_val_{white_noise_dataset_idx}.npy"
        )
        config["train_target_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_output_train_{white_noise_dataset_idx}.npy"
        )
        config["val_target_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_states_val_{white_noise_dataset_idx}.npy"
        )
        config["train_indices_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_indices_train_{white_noise_dataset_idx}.npy"
        )
        config["val_indices_file"] = (
            PHYSICS_DATA_DIR
            / "white_noise"
            / f"{config['white_noise_dataset_size']}-examples"
            / f"white_noise_indices_val_{white_noise_dataset_idx}.npy"
        )

    # Setup wandb config
    config["wandb_project"] = f"physics-pretrain-{config['predict_type']}"
    config["wandb_entity"] = "petergchang"
    # Override only if run_name is not provided
    if config["wandb_run_name"] == "default":
        config["wandb_run_name"] = run_name or "gpt"
    if config["predict_type"] == "white_noise":
        config["no_wandb"] = True

    # Set target callback
    target_callback = None
    loss_name = None

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
        target_callback=target_callback,
        loss_name=loss_name,
    )

    # For white noise models or acceleration magnitude, generate and save extrapolations
    if config["predict_type"] == "white_noise":
        ext_dir = PHYSICS_EXT_DIR / config["predict_type"]
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
    args = parse_physics_args()
    config = load_config(args)

    # Setup pretrained checkpoint directory. Default is scratch.
    if config["pretrained"] == "scratch":
        pretrained_ckpt_dir = (
            PHYSICS_CKPT_DIR / config["model_type"] / config["predict_type"]
        )
    else:
        pretrained_ckpt_dir = (
            PHYSICS_CKPT_DIR / config["model_type"] / config["pretrained"]
        )

    if config["predict_type"] == "white_noise":
        if config["white_noise_dataset_idx_lower"] is None or config["white_noise_dataset_idx_upper"] is None:
            idx_range = range(config["num_white_noise_datasets"])
        else:
            idx_range = range(config["white_noise_dataset_idx_lower"], config["white_noise_dataset_idx_upper"])
        for dataset_idx in idx_range:
            logger.info(f"[Training on white noise dataset {dataset_idx}]")

            # Build checkpoint path
            ckpt_name = f"{config['pretrained']}_pt_{config['predict_type']}"
            ckpt_name += f"_idx_{dataset_idx}_transfer"

            save_ckpt_dir = PHYSICS_CKPT_DIR / config["model_type"] / ckpt_name

            # Build run name
            run_name = (
                f"{config['white_noise_dataset_size']}_examples_"
                f"{config['max_iters']}_iters_batch_{dataset_idx}"
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
                PHYSICS_CKPT_DIR / config["model_type"] / config["predict_type"]
            )
        else:
            save_ckpt_dir = (
                PHYSICS_CKPT_DIR
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
