import copy
import json
import logging
import math
import os
import time
import warnings
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
import tqdm
import wandb

from inductivebiasprobes import ModelConfig, Model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _dt_rank_type(value):
    try:
        return int(value)
    except ValueError:
        return value


def add_common_args(parser):
    """Add common arguments to the parser."""
    # Config
    parser.add_argument("--config", type=str, help="Config file name, excluding .yaml")

    # Model architecture
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt",
        choices=["gpt", "mamba", "mamba2", "rnn", "lstm"],
    )

    # GPT architecture
    parser.add_argument("--n_embd", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, default=None, help="Number of layers")
    parser.add_argument(
        "--bias", type=bool, default=None, help="Use bias in Transformer"
    )
    parser.add_argument(
        "--n_head", type=int, default=None, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    # Mamba architecture
    parser.add_argument(
        "--dt_rank",
        type=_dt_rank_type,
        default=None,
        help="Rank of diffusion tensor",
    )
    parser.add_argument("--d_state", type=int, default=None, help="State dimension (N)")
    parser.add_argument(
        "--expand_factor", type=int, default=None, help="Expansion factor (E)"
    )
    parser.add_argument(
        "--d_conv", type=int, default=None, help="Convolution dimension"
    )
    parser.add_argument("--dt_min", type=float, default=None, help="Min diffusion time")
    parser.add_argument("--dt_max", type=float, default=None, help="Max diffusion time")
    parser.add_argument(
        "--dt_init",
        type=str,
        default=None,
        choices=["random", "constant"],
        help="Diffusion time initialization",
    )
    parser.add_argument(
        "--dt_scale", type=float, default=None, help="Diffusion time scaling"
    )
    parser.add_argument(
        "--dt_init_floor",
        type=float,
        default=None,
        help="Diffusion time init floor",
    )
    parser.add_argument(
        "--rms_norm_eps",
        type=float,
        default=None,
        help="RMS normalization epsilon",
    )
    parser.add_argument("--conv_bias", type=bool, default=None, help="Use conv bias")
    parser.add_argument(
        "--inner_layernorms",
        type=bool,
        default=None,
        help="Use inner layer norms",
    )
    parser.add_argument(
        "--pscan", type=bool, default=None, help="Use parallel scan mode"
    )

    # Checkpoints
    parser.add_argument(
        "--pretrained", default="scratch",# choices=["scratch", "next_token", "state"]
    )

    # White noise dataset
    parser.add_argument("--white_noise_dataset_size", type=int, default=100)
    parser.add_argument("--num_white_noise_datasets", type=int, default=100)

    # Training parameters
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Evaluation interval (i.e., evaluate every N iterations)",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=10,
        help="Number of iterations for evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=6e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_iters", type=int, default=60000, help="Maximum number of iterations"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument("--decay_lr", action="store_true", help="Decay learning rate")
    parser.add_argument(
        "--warmup_iters", type=int, default=2000, help="Learning rate warmup iterations"
    )
    parser.add_argument(
        "--lr_decay_iters",
        type=int,
        default=60000,
        help="Learning rate decay iterations",
    )
    parser.add_argument(
        "--min_lr", type=float, default=6e-5, help="Minimum learning rate"
    )
    parser.add_argument("--backend", default="nccl", help="Distributed backend")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Data type",
    )

    # Logging and compiling
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Logging interval (i.e., log every N iterations)",
    )
    parser.add_argument(
        "--always_save_checkpoint", action="store_true", help="Always save checkpoint"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", default="default", help="wandb project name")
    parser.add_argument("--wandb_run_name", default="default", help="wandb run name")
    parser.add_argument("--wandb_entity", default="entity", help="wandb entity name")
    parser.add_argument(
        "--no_compile", action="store_true", help="Don't compile the model"
    )
    parser.add_argument("--save_loss", action="store_true")
    parser.add_argument("--resume_from_last_ckpt", action="store_true")
    parser.add_argument("--plot_trajectory", action="store_true")
    return parser


def setup_training_environment(config, ckpt_dir, save_checkpoints=True):
    ddp = int(os.environ.get("RANK", -1)) != -1
    new_config = copy.deepcopy(config)  # Avoid modifying the original config
    if ddp:
        init_process_group(backend=new_config["backend"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        new_config["device"] = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(new_config["device"])
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        new_config["gradient_accumulation_steps"] //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    if master_process and save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    new_config["device_type"] = (
        "cuda" if new_config["device"].startswith("cuda") else "cpu"
    )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[new_config["dtype"]]

    return ddp, master_process, ptdtype, new_config


def get_sequential_batch(
    split,
    config,
    paired=False,
    full_split=False,
):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data_file = config[f"{split}_file"]
    data = np.load(data_file, mmap_mode="r")
    if full_split:
        ix = torch.arange(len(data))
    else:
        if split == 'train':
            # NOTE: This is a little dangerous, but it lets us use the same
            # config across the same dataset but different number of 
            # force vector training examples.
            num_data_points = config["num_data_points"]
        else:
            num_data_points = len(data)
        if num_data_points < config["batch_size"]:  
            ix = torch.arange(num_data_points)
        else:
            ix = torch.randint(num_data_points, (config["batch_size"],))

    # Avoid PyTorch warning by copying the array to make it writable before converting to tensor
    x = torch.stack(
        [
            torch.from_numpy(np.array(data[i, : config["block_size"]])).clone()
            for i in ix
        ]
    )
    if paired:
        target_file = config[f"{split}_target_file"]
        target_data = np.load(target_file, mmap_mode="r")
        y = torch.stack(
            [
                torch.from_numpy(
                    np.array(target_data[i, : config["block_size"]])
                ).clone()
                for i in ix
            ]
        )
    else:
        y = torch.stack(
            [
                torch.from_numpy(
                    np.array(data[i, 1 : config["block_size"] + 1])
                ).clone()
                for i in ix
            ]
        )
    if "white_noise" in config["predict_type"] and split == "train":
        # On training data, we mask out columns that aren't used for training.
        column_indices = np.load(config[f"{split}_indices_file"], mmap_mode="r")
        unmasked_column_indices = np.array([column_indices[i] for i in ix])
        # Ensure y is of a type that supports advanced indexing (e.g., long)
        y = y.long()
        unmasked_column_indices = torch.from_numpy(unmasked_column_indices).long()
        y_masked = torch.full_like(y, fill_value=config["mask_id"])
        y_masked[torch.arange(len(y)), unmasked_column_indices] = y[
            torch.arange(len(y)), unmasked_column_indices
        ]
        y = y_masked
    if config["use_float_x"]:
        x = x.float()
    else:
        x = x.long()
    if config["use_float_y"]:
        y = y.float()
    else:
        y = y.long()
    return x, y


def get_batch(
    split,
    config,
    full_split=False,
):
    x, y = get_sequential_batch(
        split,
        config,
        paired=config["predict_type"]
        in (
            "state",
            "force_magnitude",
            "force_vector",
            "balance-black",
            "edges",
            "majority",
            "parity",
        )
        or "white_noise" in config["predict_type"],
        full_split=full_split,
    )
    device, device_type = config["device"], config["device_type"]
    if device_type == "cuda":
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(iter_num, config):
    """Get learning rate based on current iteration."""
    if not config["decay_lr"]:
        return config["learning_rate"]

    learning_rate = float(config["learning_rate"])
    min_lr = float(config["min_lr"])
    warmup_iters = int(config["warmup_iters"])
    lr_decay_iters = int(config["lr_decay_iters"])

    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    if iter_num > lr_decay_iters:
        return min_lr

    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def estimate_loss(
    model, config, loss_callback=None, target_callback=None, loss_name=None
):
    """Estimate loss on train and validation sets."""
    model.eval()
    out = {}
    # First for train.
    batch_size = min(config["batch_size"], config["num_data_points"])
    losses = torch.zeros(config["eval_iters"] * batch_size)
    for k in range(config["eval_iters"]):
        X, Y = get_batch("train", config)
        with torch.no_grad():
            output, loss = model(X, Y, target_callback, loss_name=loss_name)
        if loss_callback is not None:
            loss = loss_callback(output, Y)
        start_idx = k * batch_size
        end_idx = (k + 1) * batch_size
        losses[start_idx:end_idx] = loss
    out["train"] = losses

    # Process validation in batches to avoid memory issues
    val_losses = []
    val_eval_iters = config.get("val_eval_iters", config["eval_iters"])
    for k in range(val_eval_iters):
        X, Y = get_batch("val", config)
        with torch.no_grad():
            output, loss = model(X, Y, target_callback, loss_name=loss_name)
        if loss_callback is not None:
            loss = loss_callback(output, Y)
        val_losses.append(loss.cpu())

    # Concatenate all validation losses
    if val_losses:
        # If loss is a scalar, stack; otherwise, concatenate
        if val_losses[0].dim() == 0:
            out["val"] = torch.stack(val_losses).numpy()
        else:
            out["val"] = torch.cat(val_losses).numpy()
    else:
        out["val"] = np.array([])
    model.train()
    return out


def init_model(config, ckpt_dir, ddp):
    """
    Initialize or load a model based on configuration.

    Args:
        config: Configuration dictionary containing model parameters
        ckpt_dir: Directory for checkpoints
        ddp: Whether to use DistributedDataParallel

    Returns:
        A tuple of (model, iter_num, current_epoch, best_val_loss, optimizer, scaler)
    """
    # Set up model configuration
    model_type = config["model_type"]
    if model_type == "gpt":
        model_args = init_gpt_config(config)
    elif model_type in ("mamba", "mamba2", "rnn", "lstm"):
        model_args = init_ssm_config(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Calculate iterations per epoch
    dataset_size = config["num_data_points"]
    effective_batch_size = (
        config["batch_size"]
        * config["gradient_accumulation_steps"]
        * (int(os.environ["WORLD_SIZE"]) if ddp else 1)
    )
    iters_per_epoch = dataset_size // effective_batch_size

    # Check for existing checkpoint
    fixed_configs = {
        "model_type",
        "n_embd",
        "n_layer",
        "bias",
        "input_dim",
        "block_size",
        "input_vocab_size",
        "n_head",
        "dropout",
        "dt_rank",
        "d_state",
        "expand_factor",
        "d_conv",
        "dt_min",
        "dt_max",
        "dt_init",
        "dt_scale",
        "dt_init_floor",
        "rms_norm_eps",
        "conv_bias",
        "inner_layernorms",
    }
    mutable_configs = {
        "output_dim",
        "mask_id",
        "output_vocab_size",
        "pscan",
        "use_cuda",
    }
    all_configs = fixed_configs | mutable_configs
    if config.get("resume_from_last_ckpt"):
        ckpt_path = ckpt_dir / "last_ckpt.pt"
    else:
        ckpt_path = ckpt_dir / "ckpt.pt"
    if ckpt_path.exists() and (
        config.get("resume_from_last_ckpt") or config.get("pretrained") != "scratch"
    ):
        logging.info(f"Resuming training from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=config["device"])
        checkpoint_model_args = checkpoint["model_args"]

        # Update model args from checkpoint
        load_model_args = {}
        for k in all_configs:
            if k in checkpoint_model_args:
                if (
                    k in fixed_configs
                    and k in model_args
                    and checkpoint_model_args[k] != model_args[k]
                ):
                    raise ValueError(f"Checkpoint configuration mismatch for {k}")
                load_model_args[k] = checkpoint_model_args[k]

        # Initialize model with checkpoint configuration
        model_config = ModelConfig(**model_args)
        load_model_config = ModelConfig(**load_model_args)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")
            model = Model(load_model_config)

        # Load state dict
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.config = model_config

        # Handle transfer learning case
        if config.get("pretrained") in (config.get("predict_type"), "scratch"):
            iter_num = checkpoint["iter_num"]
            current_epoch = checkpoint.get("epoch", iter_num // iters_per_epoch)
            best_val_loss = checkpoint["best_val_loss"]
            optimizer_state = checkpoint["optimizer"]
            logging.info(f"Resuming from iteration {iter_num} (epoch {current_epoch})")
        else:
            logging.info("Transfer learning from a loaded pretrained model...")
            iter_num = 0
            current_epoch = 0
            best_val_loss = float("inf")
            optimizer_state = None

            # Reset output layer for transfer learning
            model.reset_output_head()
    else:
        if config.get("pretrained") != "scratch":
            raise ValueError(
                f"Checkpoint not found at {ckpt_path} for pretrained model: {config['pretrained']}"
            )
        logging.info("Initializing a new model from scratch")
        model_config = ModelConfig(**model_args)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")
            model = Model(model_config)
        iter_num = 0
        current_epoch = 0
        best_val_loss = float("inf")
        optimizer_state = None
    model = model.to(config["device"])

    if config["no_compile"] or "mamba" in config["model_type"]:
        import torch._dynamo as dynamo

        dynamo.config.disable = True
    else:
        logging.info("Compiling model...")
        model = torch.compile(model)

    # Setup DDP if needed
    base_model = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(config["device"].split(":")[-1])]
        )

    # Setup optimizer and scaler
    optimizer = base_model.configure_optimizers(
        config["weight_decay"],
        config["learning_rate"],
        (config["beta1"], config["beta2"]),
        config["device"],
    )
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    scaler = torch.amp.GradScaler(enabled=(config["dtype"] == "float16"))
    config.update(model_args)

    return model, config, iter_num, current_epoch, best_val_loss, optimizer, scaler


def train(
    model,
    optimizer,
    scaler,
    config,
    ddp,
    master_process,
    ptdtype,
    iter_num,
    current_epoch,
    best_val_loss,
    ckpt_dir,
    save_checkpoints=True,
    callback_fn=None,
    callback_interval=None,
    callback_dir=None,
    loss_callback=None,
    loss_callback_name=None,
    target_callback=None,
    loss_name=None,
):
    """Main training loop.

    Args:
        model: The model to train
        optimizer: The optimizer to use
        scaler: The gradient scaler for mixed precision training
        config: Configuration dictionary
        ddp: Whether using distributed data parallel
        master_process: Whether this is the master process
        ptdtype: PyTorch data type
        iter_num: Current iteration number
        current_epoch: Current epoch number
        best_val_loss: Best validation loss so far
        ckpt_dir: Directory to save checkpoints
        save_checkpoints: Whether to save checkpoints
        callback_fn: Optional callback function to call during training
            The callback should take (model, val_loader, config, iter_num) as arguments
        callback_interval: How often to call the callback function (in iterations)
        callback_dir: Directory to save callback results
    """
    t0 = time.time()
    raw_model = model.module if ddp else model
    running_mfu = -1.0

    # Calculate iterations per epoch
    dataset_size = config["num_data_points"]
    effective_batch_size = (
        config["batch_size"]
        * config["gradient_accumulation_steps"]
        * (int(os.environ["WORLD_SIZE"]) if ddp else 1)
    )
    iters_per_epoch = max(dataset_size // effective_batch_size, 1)

    train_loss, val_loss = 0.0, 0.0
    pbar = tqdm.trange(
        config["max_iters"],
        desc=f"Epoch {current_epoch}, train loss {train_loss:.7f}, val loss {val_loss:.7f}",
    )
    callback_results = None

    for iter_num in pbar:
        current_epoch = iter_num // iters_per_epoch
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if (iter_num + 1) % config["eval_interval"] == 0 and master_process:
            losses = estimate_loss(
                model, config, target_callback=target_callback, loss_name=loss_name
            )
            if loss_callback is not None and loss_callback_name is not None:
                callback_losses = estimate_loss(model, config, loss_callback)
            train_loss, val_loss = losses["train"].mean(), losses["val"].mean()

            if config["plot_trajectory"]:
                evaluate_trajectory_prediction(model, config, iter_num)

            if val_loss < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = val_loss
                if iter_num > 0 and save_checkpoints:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": config,
                        "iter_num": iter_num,
                        "epoch": current_epoch,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    torch.save(checkpoint, ckpt_dir / "ckpt.pt")
                    # Save best_val_loss

                    np.save(ckpt_dir / "best_val_loss.npy", np.array(losses["val"]))
                    if loss_callback is not None and loss_callback_name is not None:
                        np.save(
                            ckpt_dir / f"best_val_{loss_callback_name}.npy",
                            np.array(callback_losses["val"]),
                        )
            # if iter_num == config["max_iters"] - 1 and save_checkpoints:
            if save_checkpoints:
                # Save each checkpoint
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": config,
                    "iter_num": iter_num,
                    "epoch": current_epoch,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                torch.save(checkpoint, ckpt_dir / "last_ckpt.pt")

            if not config["no_wandb"]:
                logging.info(
                    f"Logging to wandb with iter {iter_num} and val loss {val_loss}"
                )
                wandb.log(
                    {
                        "iter": iter_num,
                        "epoch": current_epoch,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    }
                )

        # Forward and backward pass
        X, Y = get_batch(split="train", config=config)

        for micro_step in range(config["gradient_accumulation_steps"]):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == config["gradient_accumulation_steps"] - 1
                )
            with nullcontext() if config["device"] == "cpu" else torch.amp.autocast(
                device_type="cuda", dtype=ptdtype
            ):
                _, loss = model(X, Y, target_callback, loss_name=loss_name)
                loss = loss.mean()
                loss = loss / config["gradient_accumulation_steps"]
            scaler.scale(loss).backward()

        if config["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % config["log_interval"] == 0 and master_process:
            lossf = loss.item() * config["gradient_accumulation_steps"]
            if iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    config["batch_size"] * config["gradient_accumulation_steps"], dt
                )
                if mfu is not None:
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
            train_loss = lossf

        pbar.set_description(
            f"Epoch {current_epoch}, train loss {train_loss:.7f}, val loss {val_loss:.7f}"
        )

    if callback_dir is not None and callback_results:
        callback_dir.mkdir(parents=True, exist_ok=True)
        with open(callback_dir / "callback_results.json", "w") as f:
            json.dump(callback_results, f, indent=4)

    if ddp:
        destroy_process_group()


def update_config(config, flag_config):
    valid_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    for key, value in flag_config.items():
        if value is not None and key in valid_fields:
            config[key] = value
    return config


def init_gpt_config(config):
    """
    Initialize a GPT model configuration.

    Args:
        config: Configuration dictionary to override defaults

    Returns:
        Configuration dictionary for GPT model
    """
    defaults = {
        "model_type": "gpt",
        "n_embd": 768,
        "n_layer": 12,
        "bias": True,
        "n_head": 12,
        "dropout": 0.0,
    }
    defaults = update_config(defaults, config)

    return defaults


def init_ssm_config(config):
    """
    Initialize a State Space Model (Mamba/RNN/LSTM) configuration.

    Args:
        config: Configuration dictionary to override defaults

    Returns:
        Configuration dictionary for SSM model
    """
    model_type = config["model_type"]
    if model_type not in ("mamba", "mamba2", "rnn", "lstm"):
        raise ValueError(f"Unsupported SSM model type: {model_type}")

    defaults = {
        "n_embd": 768,
        "bias": True,
        "dt_rank": "auto",
        "d_state": 16,
        "expand_factor": 2,
        "d_conv": 4,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "dt_init": "random",
        "dt_scale": 1.0,
        "dt_init_floor": 1e-4,
        "rms_norm_eps": 1e-5,
        "conv_bias": True,
        "inner_layernorms": True,
        "pscan": True,
        "use_cuda": True,
    }

    if model_type in ("mamba", "mamba2"):
        mamba_defaults = {
            "n_layer": 24,
        }
        defaults.update(mamba_defaults)
        if config["n_layer"] is not None:  # Double the number of layers for mamba
            config["n_layer"] *= 2
    else:
        rnn_defaults = {
            "n_layer": 6,
        }
        defaults.update(rnn_defaults)

    defaults = update_config(defaults, config)

    return defaults


def evaluate_trajectory_prediction(model, config, iter_num=None):
    """
    Evaluate model by predicting future trajectory steps and log to wandb.
    Takes a random trajectory, uses first 100 steps as context, and predicts 50 steps ahead.
    """
    # Sample the solar system trajectory
    X, _ = get_batch(split="test", config=config, full_split=True)
    # x = X[0] # Uncomment to sample the solar system trajectory
    sample_ind = np.random.randint(0, X.shape[0])
    x = X[sample_ind]

    # Use first 300 steps as prefix
    prefix_length = 500
    prefix = x[:prefix_length].reshape(1, prefix_length, -1)

    # Start with prefix and iteratively predict next steps
    curr_obs = prefix.to(config["device"])
    completed_trajectory = []
    num_steps_out = 500

    # Get ground truth for comparison
    true_trajectory = x[prefix_length : prefix_length + num_steps_out].cpu().numpy()

    with torch.no_grad():
        for _ in range(num_steps_out):
            pred = model(curr_obs)

            # --- Discrete case: logits need to be split per-coordinate --------
            if config.get("output_vocab_size") is not None:
                logits = pred[:, -1, :]
                vocab = config["output_vocab_size"]
                # Assume coords are concatenated as [x-logits, y-logits, z-logits]
                logits_per_coord = [
                    logits[:, i * vocab : (i + 1) * vocab]
                    for i in range(config["input_dim"])
                ]
                next_coords = [
                    torch.argmax(logit, dim=-1) for logit in logits_per_coord
                ]
                # shape (batch, input_dim)
                next_point = torch.stack(next_coords, dim=-1).unsqueeze(1)
            else:
                # Continuous output
                next_point = pred[:, -1].reshape(1, 1, -1)
            curr_obs = torch.cat([curr_obs, next_point], dim=1)
            completed_trajectory.append(next_point.cpu().numpy().reshape(-1))

    # Convert to numpy arrays (undo centering shift if discrete)
    if config.get("output_vocab_size") is not None:
        shift = config["output_vocab_size"] // 2
        prefix_np = prefix.cpu().numpy()[0] - shift
        completed_trajectory = np.array(completed_trajectory) - shift
        true_trajectory = true_trajectory - shift
    else:
        prefix_np = prefix.cpu().numpy()[0]
        completed_trajectory = np.array(completed_trajectory)

    fig = plt.figure(figsize=(15, 15))
    plt.scatter(prefix_np[:, 0], prefix_np[:, 1], c="r", label="Context (500 steps)")
    plt.scatter(
        completed_trajectory[:, 0],
        completed_trajectory[:, 1],
        c="b",
        label="Prediction (500 steps)",
    )
    plt.scatter(
        true_trajectory[:, 0], true_trajectory[:, 1], c="g", label="Ground Truth"
    )
    plt.legend()
    plt.title(f"Trajectory Prediction at Iteration {iter_num}")
    plt.xlabel("X position")
    plt.ylabel("Y position")

    fig = plt.gcf()  # Get the current figure
    if not config["no_wandb"]:
        wandb.log({"trajectory_prediction": wandb.Image(fig), "iter": iter_num})
        plt.close(fig)


def fit_mlp_full_batch(
    model,
    loss_fn,
    optimizer,
    X_train,
    Y_train,
    X_val,
    Y_val=None,
    epochs=100,
):
    # Cast to torch float32
    X_train, Y_train, X_val = (
        torch.from_numpy(X_train).float(),
        torch.from_numpy(Y_train).float(),
        torch.from_numpy(X_val).float(),
    )
    if Y_val is not None:
        Y_val = torch.from_numpy(Y_val).float()

    # Training loop
    pbar = tqdm.tqdm(range(epochs), desc="Training MLP:")
    for _ in pbar:
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        if Y_val is not None:
            val_loss = loss_fn(val_predictions, Y_val)
            logger.info(f"Validation loss: {val_loss.item():.4f}")
    return val_predictions


def generate_and_save_extrapolations(model, config, ext_dir, ext_idx_dir):
    """Generate and save model extrapolations."""
    model.eval()
    with torch.no_grad():
        all_obs, all_states = get_batch(split="val", config=config, full_split=True)
        all_predictions = model(all_obs)
    ext_idx_dir.mkdir(parents=True, exist_ok=True)

    # If states.npy doesn't exist in ext_dir, save states
    if not (ext_dir / "states.npy").exists():
        np.save(ext_dir / "states.npy", all_states.cpu().numpy())

    # If obs.npy doesn't exist in ext_dir, save obs
    if not (ext_dir / "obs.npy").exists():
        np.save(ext_dir / "obs.npy", all_obs.cpu().numpy())

    # Save extrapolations
    np.save(ext_idx_dir / "extrapolations.npy", all_predictions.cpu().numpy())
