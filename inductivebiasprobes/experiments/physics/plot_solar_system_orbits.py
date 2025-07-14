import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import os
import tqdm
import yaml
from inductivebiasprobes import ModelConfig, Model
from inductivebiasprobes.paths import PHYSICS_DATA_DIR, PHYSICS_CONFIG_DIR

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
sns.set()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_length", type=int, default=50)
    parser.add_argument("--pretrained", type=str, default="next_token")
    return parser.parse_args()


def load_model(pretrained):
    base_dir = Path(__file__).resolve().parent.parent.parent
    ckpt_path = base_dir / "checkpoints" / "physics" / "gpt" / "next_token" / "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    checkpoint_model_args = checkpoint["model_args"]

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
    load_model_args = {}
    for k in all_configs:
        if k in checkpoint_model_args:
            load_model_args[k] = checkpoint_model_args[k]

    model_config = ModelConfig(**load_model_args)
    model = Model(model_config)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.config = model_config
    return model


def main():
    args = parse_args()
    model = load_model(args.pretrained)
    model.eval()
    device = "cuda"
    model.to(device)

    with (PHYSICS_CONFIG_DIR / ("ntp_config.yaml")).open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data = np.load(PHYSICS_DATA_DIR / f"obs_solar_system_single_sequence.npy")
    x = torch.from_numpy(data).long()
    
    prefix_length = 300 
    num_steps_out = len(x) - prefix_length

    prefix = x[:prefix_length].reshape(1, prefix_length, -1).to(device)
    true_trajectory = x[prefix_length:prefix_length+num_steps_out].cpu().numpy()

    curr_obs = prefix
    completed_trajectory = []
    bar = tqdm.tqdm(range(num_steps_out))
    for i in bar:
        with torch.no_grad():
            pred = model(curr_obs.to(device))
            logits = pred[:, -1, :]
            logits_per_coord = [
                logits[:, i * config["output_vocab_size"] : (i + 1) * config["output_vocab_size"]]
                for i in range(config["input_dim"])
            ]
            next_coords = [
                torch.argmax(logit, dim=-1) for logit in logits_per_coord
            ]
            next_point = torch.stack(next_coords, dim=-1).unsqueeze(1)
            curr_obs = torch.cat([curr_obs, next_point], dim=1)
            completed_trajectory.append(next_point.cpu().numpy().reshape(-1))
    completed_trajectory = np.array(completed_trajectory)

    prefix = prefix.cpu().numpy()[0] - config['output_vocab_size'] // 2
    prefix = prefix[1:]
    completed_trajectory = np.array(completed_trajectory) - config['output_vocab_size'] // 2
    true_trajectory = true_trajectory - config['output_vocab_size'] // 2

    prefix_and_completed = np.concatenate([prefix, completed_trajectory], axis=0)
    prefix_and_true = np.concatenate([prefix, true_trajectory], axis=0)

    # Reshape so that each planet is a row
    num_planets = 9  # includes sun.
    num_rows = len(prefix_and_true)
    shared_timestep_num = len(prefix) // num_planets
    num_rows_mod_num_planets = num_rows % num_planets
    num_rows_div_num_planets = num_rows - num_rows_mod_num_planets
    true_subset = prefix_and_true[:num_rows_div_num_planets, :]
    true_by_planet = true_subset.reshape(num_planets, -1, 2, order='F')
    pred_subset = prefix_and_completed[:num_rows_div_num_planets, :]
    pred_by_planet = pred_subset.reshape(num_planets, -1, 2, order='F')

    # Plot the orbits
    num_planets, num_timesteps, _ = pred_by_planet.shape
    num_planets -= 1 # Don't plot sun

    # Planet colors based on true appearance
    planet_colors = [
        "#8C8680",  # Mercury - gray
        "#C9A87C",  # Venus - golden brown
        "#2A6DDF",  # Earth - blue
        "#C1440E",  # Mars - reddish brown
        "#DCA04D",  # Jupiter - orange/tan
        "#E7CDAB",  # Saturn - pale gold
        "#B2D9E3",  # Uranus - pale cyan
        "#3467CC",  # Neptune - deep blue
        "#9F7358",  # Pluto (if needed) - brownish
    ]

    # Relative planet sizes (relative to Earth = 1)
    # Values are approximated from NASA data
    planet_sizes = [
        0.38,    # Mercury - ~0.38x Earth's size
        0.95,    # Venus - ~0.95x Earth's size
        1.0,     # Earth 
        0.53,    # Mars - ~0.53x Earth's size
        11.2,    # Jupiter - ~11.2x Earth's size
        9.4,     # Saturn - ~9.4x Earth's size 
        4.0,     # Uranus - ~4.0x Earth's size
        3.9,     # Neptune - ~3.9x Earth's size
        0.18,    # Pluto (if needed) - ~0.18x Earth's size
    ]

    # Base size (Earth) in points
    base_size = 10
    planet_marker_sizes = [base_size * s**2 for s in planet_sizes[:num_planets]]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Predicted vs. True Planet Trajectories")

    # First: previously-seen ("observed") portion
    for p in range(num_planets):
        ax.plot(
            true_by_planet[p, :shared_timestep_num, 0],
            true_by_planet[p, :shared_timestep_num, 1],
            ls="--", lw=1.2, alpha=0.3, color=planet_colors[p],
        )

    # Containers for animated artists
    pred_lines, true_lines = [], []
    pred_pts, true_pts = [], []

    for p in range(num_planets):
        ln_pred, = ax.plot([], [], lw=2, color=planet_colors[p])          # solid → prediction
        ln_true, = ax.plot([], [], lw=2, color=planet_colors[p], ls=":")  # dotted → truth
        
        # Use planet-specific marker sizes
        pt_pred = ax.scatter([], [], s=planet_marker_sizes[p], color=planet_colors[p])
        pt_true = ax.scatter([], [], s=planet_marker_sizes[p], facecolors="none", 
                             edgecolors=planet_colors[p], alpha=0.5)
        
        pred_lines.append(ln_pred)
        true_lines.append(ln_true)
        pred_pts.append(pt_pred)
        true_pts.append(pt_true)

    # nice limits
    all_x = np.concatenate([pred_by_planet[...,0], true_by_planet[...,0]])
    all_y = np.concatenate([pred_by_planet[...,1], true_by_planet[...,1]])
    pad = 0.1 * max(np.ptp(all_x), np.ptp(all_y))
    ax.set_xlim(all_x.min()-pad, all_x.max()+pad)
    ax.set_ylim(all_y.min()-pad, all_y.max()+pad)

    # legend (one handle per meaning, not per planet)
    proxy_pred = plt.Line2D([],[], lw=2, color="k", label="prediction")
    proxy_true = plt.Line2D([],[], lw=2, ls=":", color="k", label="ground truth")
    proxy_obs = plt.Line2D([],[], lw=1.2, ls="--", color="k", alpha=0.3, label="observed")

    # Create two legends - one for line styles, one for planets
    line_legend = ax.legend(handles=[proxy_pred, proxy_true, proxy_obs], loc="lower right")
    ax.add_artist(line_legend)  # Add first legend

    # -------------------------------------------------
    # Animation functions
    # -------------------------------------------------
    def init():
        for line in pred_lines + true_lines:
            line.set_data([], [])
        for pt in pred_pts + true_pts:
            pt.set_offsets(np.empty((0,2)))
        return pred_lines + true_lines + pred_pts #+ true_pts

    def update(f):
        """Frame f animates timestep t = shared_timestep_num + f"""
        t = shared_timestep_num + f
        for p in range(num_planets):
            # slice from shared point inclusive → current frame
            xs_pred, ys_pred = pred_by_planet[p, shared_timestep_num:t+1].T
            xs_true, ys_true = true_by_planet[p, shared_timestep_num:].T
            xs_true_moving, ys_true_moving = true_by_planet[p, shared_timestep_num:t+1].T

            pred_lines[p].set_data(xs_pred, ys_pred)
            true_lines[p].set_data(xs_true, ys_true)

            pred_pts[p].set_offsets([xs_pred[-1], ys_pred[-1]])
            true_pts[p].set_offsets([xs_true_moving[-1], ys_true_moving[-1]])
        return pred_lines + true_lines + pred_pts #+ true_pts

    frames = num_timesteps - shared_timestep_num
    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=50, blit=True
    )

    os.makedirs('figs', exist_ok=True)
    ani.save('figs/solar_system_orbit_predictions.gif', writer='pillow', fps=20)
    plt.close()



if __name__ == "__main__":
    main()