import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import tqdm

from inductivebiasprobes.paths import (
    PHYSICS_CKPT_DIR,
    PHYSICS_DATA_DIR,
)
from inductivebiasprobes import Model, ModelConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a next-token-prediction (NTP) model on the planetary dataset, both in teacher-forcing and autoregressive settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt",
        choices=["gpt", "mamba", "mamba2", "rnn", "lstm"],
    )
    parser.add_argument("--predict_type", type=str, default="next_token")
    parser.add_argument(
        "--prefix_points",
        type=int,
        default=500,
        help="Prefix length used for autoregressive evaluation.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="*",
        default=[1, 5, 100],
        help="Forecast horizons (in steps) for AR evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_checkpoint(
    ckpt_path: Path, device: torch.device | str = "cpu"
) -> tuple[Model, dict]:
    """Load a checkpoint and return the instantiated model (eval mode) and its args."""
    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint["model_args"]
    if "input_dim" not in model_args and "config" in checkpoint:
        model_args = checkpoint["config"]

    # Filter out keys not part of ModelConfig (e.g., training-only params)
    allowed_keys = set(ModelConfig.__dataclass_fields__.keys())
    filtered_args = {k: v for k, v in model_args.items() if k in allowed_keys}
    missing = allowed_keys - filtered_args.keys()
    if missing:
        logger.warning(
            "ModelConfig missing expected keys from checkpoint: %s. These must be provided manually or via --model_type.",
            ", ".join(sorted(missing)),
        )

    model_config = ModelConfig(**filtered_args)
    model = Model(model_config)
    # Strip DDP prefix if necessary
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, filtered_args


def autoregressive_predict(
    model: Model,
    prefix: torch.Tensor,
    num_steps: int,
    cfg: dict,
    device: torch.device,
) -> np.ndarray:
    """Generate *num_steps* predictions autoregressively.

    The function handles both discrete-(vocab) and continuous models.

    Returns
    -------
    np.ndarray
        Array of shape (num_steps, input_dim) containing the newly-generated points
        **not** including the prefix.
    """
    assert prefix.ndim == 3  # (B, T, D)
    input_dim = cfg.get("input_dim", 3)
    output_vocab = cfg.get("output_vocab_size")

    seq = prefix.clone()  # we will keep appending to this tensor
    generated = []
    for _ in range(num_steps):
        with torch.no_grad():
            pred = model(seq)
        if output_vocab is not None:  # discrete case – logits
            logits = pred[:, -1]  # (B, D*V)
            # Split the big logits vector into per-coordinate chunks
            coord_logits = [
                logits[:, i * output_vocab : (i + 1) * output_vocab]
                for i in range(input_dim)
            ]
            next_coords = [torch.argmax(logit, dim=-1) for logit in coord_logits]
            next_point = torch.stack(next_coords, dim=-1).unsqueeze(1)
        else:  # continuous regression – assume final layer gives value directly
            next_point = pred[:, -1].reshape(1, 1, input_dim)
        seq = torch.cat([seq, next_point], dim=1)
        generated.append(next_point.squeeze(0).cpu().numpy())
    return np.concatenate(generated, axis=0)  # (num_steps, D)


def idx_to_coordinate_2d(
    idx: np.ndarray,
    num_bins: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Convert 2-D bin indices back to continuous coordinates.

    The x-, y-coordinates may use different [min,max] ranges, matching the
    discretisation performed in ``generate_data``.
    """

    if idx.shape[-1] != 2:
        raise ValueError("idx array must have last dimension 2 (x,y)")

    # Ensure float conversion for arithmetic
    idx_f = idx.astype(float)

    width_x = (x_max - x_min) / num_bins
    width_y = (y_max - y_min) / num_bins

    out = np.empty_like(idx_f, dtype=float)
    out[..., 0] = x_min + (idx_f[..., 0] + 0.5) * width_x
    out[..., 1] = y_min + (idx_f[..., 1] + 0.5) * width_y

    return out


def _to_continuous(array: np.ndarray, cfg: dict) -> np.ndarray:
    """Convert discretised indices to continuous positions when necessary."""
    if cfg.get("input_vocab_size") is None:
        return array.astype(float)

    V = cfg["input_vocab_size"]
    return idx_to_coordinate_2d(
        array,
        V,
        cfg.get("traj_xy_min", -1.5),  # Fallbacks – real values live in generate_data
        cfg.get("traj_xy_max", 1.5),
        cfg.get("traj_xy_min", -1.5),
        cfg.get("traj_xy_max", 1.5),
    )


def _model_forward_teacher_forcing(
    model: Model, x_tensor: torch.Tensor, cfg: dict
) -> np.ndarray:
    """Run the model in teacher-forcing mode and return next-step predictions."""
    # Predict the next token given the *full* prefix up to T-1.
    with torch.no_grad():
        preds = model(x_tensor[:, :-1])  # (B, T-1, D*V) or (B, T-1, D)

    if cfg.get("output_vocab_size") is not None:  # logits –> indices –> coords
        V = cfg["output_vocab_size"]
        D = cfg.get("output_dim", 2)
        logits = [preds[:, :, i * V : (i + 1) * V] for i in range(D)]
        idx_pred = (
            torch.stack([logit.argmax(dim=-1) for logit in logits], dim=-1)
            .cpu()
            .numpy()
        )
        return _to_continuous(idx_pred, cfg)  # (B, T-1, D)
    return preds.cpu().numpy()  # (B, T-1, D)


def _prepare_tensor(array: np.ndarray, cfg: dict, device: torch.device) -> torch.Tensor:
    if cfg.get("input_vocab_size") is not None and not cfg.get("use_float_x", False):
        return torch.from_numpy(array).long().to(device)
    return torch.from_numpy(array).float().to(device)


def compute_teacher_forcing_metrics(
    model: Model,
    model_cfg: dict,
    obs_test: np.ndarray,
    obs_train: np.ndarray,
    num_bodies_arr: np.ndarray,
    device: torch.device,
) -> dict:
    """Compute teacher-forcing MSE for model and baselines."""

    if isinstance(num_bodies_arr, int):
        num_bodies_arr = np.full(obs_test.shape[0], num_bodies_arr, dtype=int)

    if obs_test.shape[0] != len(num_bodies_arr):
        raise ValueError("num_bodies_arr length must match batch size of obs_test")

    # Model prediction
    x_tensor = _prepare_tensor(obs_test, model_cfg, device)
    pred_next = _model_forward_teacher_forcing(
        model, x_tensor, model_cfg
    )  # (B, T-1, D)
    gt_next = _to_continuous(obs_test[:, 1:], model_cfg)  # (B, T-1, D)

    # Baseline – global mean (training set)
    baseline_global_mean = np.mean(_to_continuous(obs_train, model_cfg), axis=(0, 1))

    # Baseline – per-sequence mean
    seq_means = np.mean(
        _to_continuous(obs_test, model_cfg), axis=1, keepdims=True
    )  # (B,1,D)

    # Baseline – last token (persistence, same planet)
    model_errors = []
    baseline_global_errors = []
    baseline_seq_mean_errors = []
    baseline_last_errors = []
    for i, nb in enumerate(tqdm.tqdm(num_bodies_arr)):
        if obs_test.shape[1] <= nb:
            raise ValueError(
                f"Sequence too short to compute persistence baseline for trajectory {i} with num_bodies={nb}."
            )
        baseline_last_i = _to_continuous(obs_test[i, :-nb], model_cfg)
        gt_future_last_i = gt_next[i, nb - 1 :]
        model_errors.append(np.mean((pred_next[i, nb - 1 :] - gt_future_last_i) ** 2))
        baseline_global_errors.append(
            np.mean((baseline_global_mean - gt_future_last_i) ** 2)
        )
        baseline_seq_mean_errors.append(np.mean((seq_means[i] - gt_future_last_i) ** 2))
        baseline_last_errors.append(np.mean((baseline_last_i - gt_future_last_i) ** 2))

    mse_model = np.mean(model_errors)
    se_model = np.std(model_errors) / np.sqrt(len(model_errors))
    mse_baseline_global = np.mean(baseline_global_errors)
    se_baseline_global = np.std(baseline_global_errors) / np.sqrt(
        len(baseline_global_errors)
    )
    mse_baseline_seq_mean = np.mean(baseline_seq_mean_errors)
    se_baseline_seq_mean = np.std(baseline_seq_mean_errors) / np.sqrt(
        len(baseline_seq_mean_errors)
    )
    mse_baseline_last = np.mean(baseline_last_errors)
    se_baseline_last = np.std(baseline_last_errors) / np.sqrt(len(baseline_last_errors))

    return {
        "model": {
            "mse": mse_model,
            "se": se_model,
        },
        "baseline_global": {
            "mse": mse_baseline_global,
            "se": se_baseline_global,
        },
        "baseline_seq_mean": {
            "mse": mse_baseline_seq_mean,
            "se": se_baseline_seq_mean,
        },
        "baseline_last": {
            "mse": mse_baseline_last,
            "se": se_baseline_last,
        },
    }


def compute_autoregressive_metrics(
    model: Model,
    model_cfg: dict,
    obs_train: np.ndarray,
    obs_test: np.ndarray,
    prefix_len: int,
    horizons: Sequence[int],
    num_bodies_arr: np.ndarray,
    device: torch.device,
) -> dict:
    """Compute horizon-specific MSEs using autoregressive generation."""

    max_h = max(horizons)
    assert (
        prefix_len + max_h < obs_test.shape[1]
    ), "Prefix+horizon exceeds sequence length."

    if isinstance(num_bodies_arr, int):
        num_bodies_arr = np.full(obs_test.shape[0], num_bodies_arr, dtype=int)

    # Baseline – global mean (training set)
    baseline_global_mean = np.mean(_to_continuous(obs_train, model_cfg), axis=(0, 1))

    # Baseline – per-sequence mean
    seq_means = np.mean(
        _to_continuous(obs_test, model_cfg), axis=1, keepdims=True
    )  # (B,1,D)

    errors: Dict[str, Dict[int, Dict[str, Any]]] = {
        "model": {h: [] for h in horizons},
        "baseline_global": {h: [] for h in horizons},
        "baseline_seq_mean": {h: [] for h in horizons},
        "baseline_last": {h: [] for h in horizons},
    }
    for idx, traj in enumerate(tqdm.tqdm(obs_test)):  # iterate over trajectories
        prefix = traj[:prefix_len]
        x_tensor = _prepare_tensor(prefix[None, ...], model_cfg, device)  # (1, P, D)
        gen = autoregressive_predict(
            model, x_tensor, max_h, model_cfg, device
        )  # (max_h, D)
        gen_cont = _to_continuous(gen, model_cfg)
        traj_cont = _to_continuous(traj, model_cfg)
        gt_cont = traj_cont[prefix_len:]
        
        nb = int(num_bodies_arr[idx])
        for h in horizons:
            last_tok_idx = prefix_len + (h - 1) - nb * ((h // nb) + 1)
            last_tok = traj_cont[last_tok_idx]
            pred_tok = gen_cont[h - 1]
            gt_tok = gt_cont[h - 1]
            errors["model"][h].append(np.mean((pred_tok - gt_tok) ** 2))
            errors["baseline_global"][h].append(np.mean((baseline_global_mean - gt_tok) ** 2))
            errors["baseline_seq_mean"][h].append(np.mean((seq_means[idx] - gt_tok) ** 2))
            errors["baseline_last"][h].append(np.mean((last_tok - gt_tok) ** 2))

    return {k: {h: {"mse": np.mean(v).item(), "se": np.std(v) / np.sqrt(len(v)).item()} for h, v in d.items()} for k, d in errors.items()}


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load checkpoint
    ckpt_dir = PHYSICS_CKPT_DIR / args.model_type / args.predict_type
    ckpt_path = ckpt_dir / "ckpt.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "last_ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    model, model_cfg = load_checkpoint(ckpt_path, device)

    # Load data
    obs_train = np.load(PHYSICS_DATA_DIR / "obs_train.npy")
    obs_test_path = PHYSICS_DATA_DIR / "obs_test.npy"
    if not obs_test_path.exists():
        obs_test_path = PHYSICS_DATA_DIR / "obs_test_raw.npy"  # fallback
    obs_test = np.load(obs_test_path)

    # If data has leading batch dim of 1 (solar-system style), squeeze it
    if obs_train.ndim == 3:
        pass  # expected shape (B, T, D)
    if obs_test.ndim == 3:
        pass

    # Infer number of bodies per trajectory from metadata
    split_label = obs_test_path.stem[
        len("obs_") :
    ]  # crude but works with obs_test, obs_two_body_test, etc.
    num_bodies_file = PHYSICS_DATA_DIR / f"num_bodies_{split_label}.npy"
    if num_bodies_file.exists():
        num_bodies_arr = np.load(num_bodies_file).astype(int)
        logger.info(f"Loaded num_bodies metadata from {num_bodies_file}")
    else:
        logger.warning(
            "num_bodies metadata file not found. Assuming constant 10 bodies."
        )
        num_bodies_arr = np.full(obs_test.shape[0], 10, dtype=int)

    logger.info("Computing teacher-forcing metrics …")
    tf_metrics = compute_teacher_forcing_metrics(
        model,
        model_cfg,
        obs_test,
        obs_train,
        num_bodies_arr,
        device,
    )
    for k, v in tf_metrics.items():
        logger.info(f"{k}: {v['mse']:.5e} ± {v['se']:.5e}")

    logger.info("Computing autoregressive metrics …")
    ar_metrics = compute_autoregressive_metrics(
        model,
        model_cfg,
        obs_train,
        obs_test,
        args.prefix_points,
        args.horizons,
        num_bodies_arr,
        device,
    )
    for k, v in ar_metrics.items():
        for h in args.horizons:
            logger.info(f"{k} h={h}: {v[h]['mse']:.5e} ± {v[h]['se']:.5e}")

    # Save
    out_dir = PHYSICS_CKPT_DIR / args.model_type / args.predict_type / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "teacher_forcing_metrics.npz",
        **tf_metrics,
    )
    np.savez(
        out_dir / f"autoregressive_metrics_prefix_{args.prefix_points}.npz",
        **ar_metrics,
    )
    logger.info(f"Saved metrics to {out_dir}")


if __name__ == "__main__":
    main()