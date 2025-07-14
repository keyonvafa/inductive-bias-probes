import logging
from pathlib import Path

import numpy as np
import torch
import os
import yaml
from inductivebiasprobes import ModelConfig, Model
from inductivebiasprobes.paths import PHYSICS_DATA_DIR, PHYSICS_CONFIG_DIR
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
sns.set()

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PLANET_NAMES = ["Mercury", "Venus", "Earth", "Mars",
                "Jupiter", "Saturn", "Uranus", "Neptune"]
PLANET_COLORS = ["#B1B1B1", "#F5C16C", "#4EA3FF", "#C1440E",
                 "#F1E4B3", "#D8C17A", "#7FFFD4", "#4256FF"]

def load_model(checkpoint):
    base_dir = Path(__file__).resolve().parent.parent.parent
    ckpt_path = base_dir / "checkpoints" / "physics" / "gpt" / checkpoint / "ckpt.pt"
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


def undiscretize_data(data, num_bins, min_value, max_value):
    lo, hi = min_value, max_value
    width = (hi - lo) / num_bins
    # Calculate the midpoint of each bin
    undiscretized_values = lo + (data + 0.5) * width
    return undiscretized_values


def farthest_point_sampling(candidates: np.ndarray, k: int, start_idx: int | None = None):
    """
    Pick k points from `candidates` (shape [N,2]) so that the minimum pairwise
    distance inside the picked set is as large as possible, using greedy FPS.

    Parameters
    ----------
    candidates : (N,2) float array
    k          : number of points to return
    start_idx  : optional index for the first point (defaults to random)

    Returns
    -------
    idx  : (k,) int array of indices into `candidates`
    pts  : (k,2) float array of the selected coordinates
    """
    N = len(candidates)
    if k >= N:
        return np.arange(N), candidates.copy()

    # Pre-compute the full distance matrix once (saves work for moderate N)
    D = distance.cdist(candidates, candidates, metric='euclidean')

    if start_idx is None:
        start_idx = np.random.randint(N)

    chosen = [start_idx]                          # indices we already have
    d_to_set = D[start_idx].copy()                # distance of every point to nearest chosen

    for _ in range(1, k):
        next_idx = np.argmax(d_to_set)            # farthest point so far
        chosen.append(next_idx)
        # update: distance to the new set is the min of old dist and dist to new point
        d_to_set = np.minimum(d_to_set, D[next_idx])

    chosen = np.array(chosen)
    return chosen, candidates[chosen]


def add_arrows_3d(ax, x, y, u, v, *, scale=1.0, linewidth=2.0,
                  color="k", alpha=1.0, arrow_length_ratio=0.13, label=None):
    z = np.zeros_like(x)
    w = np.zeros_like(u)
    ax.quiver(x, y, z,
              u * scale, v * scale, w,
              length=1, normalize=False,
              linewidth=linewidth,
              color=color,
              arrow_length_ratio=arrow_length_ratio,
              alpha=alpha)
    if label is not None:
        ax.plot([], [], [], color=color, linewidth=linewidth, label=label)


def setup_common(ax, planet_idx, title_suffix, arrow_u, arrow_v,
                 arrow_color, arrow_label,
                 xmin, xmax, ymin, ymax, zmin, zmax,
                 force_scale, planet_sizes, sun_size,
                 obs, timesteps_to_plot,
                 draw_title=True):
    planet_legend_name = PLANET_NAMES[planet_idx] if draw_title else None
    ax.scatter(obs[timesteps_to_plot, 0],
               obs[timesteps_to_plot, 1],
               zs=0, c=PLANET_COLORS[planet_idx],
               s=planet_sizes[planet_idx],
               label=planet_legend_name)

    add_arrows_3d(ax,
                  obs[timesteps_to_plot, 0],
                  obs[timesteps_to_plot, 1],
                  arrow_u, arrow_v,
                  scale=force_scale,
                  linewidth=3 if planet_idx == 2 else 2.5, # 2.25, 1.5
                  color=arrow_color,
                  arrow_length_ratio=0.20,
                  label=arrow_label)

    ax.scatter(0, 0, 0, s=sun_size, c="gold", edgecolors="darkorange", zorder=3)
    
    if draw_title:
        from matplotlib.lines import Line2D
        earth_marker_size = np.sqrt(planet_sizes[2])
        sun_handle = Line2D([], [], marker='o', linestyle='None',
                            markersize=earth_marker_size,
                            markerfacecolor='gold',
                            markeredgecolor='darkorange',
                            label="Sun")

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor("#333333")
        pane.set_facecolor("black")
        pane.set_alpha(1.00)
        
    ax.view_init(elev=30, azim=45)

    if draw_title:
        ax.set_title(f"{PLANET_NAMES[planet_idx]} {title_suffix}",
                     color="black",
                     pad=-200, fontsize=35)

    handles, labels = ax.get_legend_handles_labels()
    custom_handles, custom_labels = [], []

    for h, l in zip(handles, labels):
        if l == PLANET_NAMES[planet_idx]:
            custom_handles.append(h)
            custom_labels.append(l)

    if draw_title:
        custom_handles.append(sun_handle)
        custom_labels.append("Sun")

    for h, l in zip(handles, labels):
        if l not in custom_labels:
            custom_handles.append(h)
            custom_labels.append(l)

    if draw_title: # just earth
        ncols = len(custom_handles)          # put every entry in one row
        leg = ax.legend(
                handles=custom_handles,
                labels=custom_labels,
                loc="upper center",          # start at the horizontal mid-point
                bbox_to_anchor=(0.5, 0.95),  # fine-tune position just above the axes
                ncol=ncols,                  # ***makes it horizontal***
                facecolor="white",
                fontsize=20,
                handlelength=1.2,
                handletextpad=0.4,
                columnspacing=0.8            # (optional) space between columns
        )

    else: # other planets
        leg = ax.legend(handles=custom_handles, labels=custom_labels,
                        loc="upper center",
                        facecolor="white",
                        fontsize=35,
                        handlelength=1.2, handletextpad=0.4)

    for txt in leg.get_texts():
        txt.set_color("black")


def make_force_pictures(all_preds, all_truth, all_obs):
    force_scale = 18 
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#444444",
        "grid.alpha": 0.4,
        "legend.framealpha": 0.8,
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    num_to_plot = 40
    smas = [0.387, 0.723, 1.000, 1.524, 5.203, 9.537, 19.189, 30.069]
    planet_sizes = {0: 100, 1: 200, 2: 300, 3: 150, 4: 300, 5: 300, 6: 250, 7: 250}

    for planet_idx in range(8):
        force = all_truth[planet_idx]
        predicted_force = all_preds[planet_idx]
        obs = all_obs[planet_idx]

        timesteps_to_plot, _ = farthest_point_sampling(obs, num_to_plot)

        orbit_scale = smas[-1] / smas[planet_idx]

        obs = orbit_scale * obs

        sun_size = 3600 if planet_idx == 2 else 800 - 700 * planet_idx / 7
        xmin, xmax = obs[:, 0].min() + 5, obs[:, 0].max() - 5
        ymin, ymax = obs[:, 1].min() + 5, obs[:, 1].max() - 5
        zmin, zmax = -25, 25

        figsize = (14, 8) if planet_idx != 2 else (20, 10)
        fig = plt.figure(figsize=figsize)
        ax_true = fig.add_subplot(1, 2, 1, projection="3d")
        ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
        
        draw_titles = (planet_idx == 2)
        if planet_idx != 2:                    # 2 → Earth
            fig.subplots_adjust(wspace=0.05,   # 0.05
                                left=0.05,     # tighten side margins a touch
                                right=0.95)
        else:
            fig.subplots_adjust(wspace=-0.25,   # -0.10
                                left=0.05,      # 0.07
                                right=0.95)     # 0.93


        setup_common(ax_true, planet_idx, "(true)",
                     force[timesteps_to_plot, 0],
                     force[timesteps_to_plot, 1],
                     "#00A0C0", "True force",
                     xmin, xmax, ymin, ymax, zmin, zmax,
                     force_scale, planet_sizes, sun_size,
                     obs, timesteps_to_plot,
                     draw_title=draw_titles)

        setup_common(ax_pred, planet_idx, "(predicted)",
                     predicted_force[timesteps_to_plot, 0],
                     predicted_force[timesteps_to_plot, 1],
                     "#FF1493", "Predicted force",
                     xmin, xmax, ymin, ymax, zmin, zmax,
                     force_scale, planet_sizes, sun_size,
                     obs, timesteps_to_plot,
                     draw_title=draw_titles)

        if planet_idx != 2:
            center_x = 0.5
            fig.text(center_x, 0.87,
                     PLANET_NAMES[planet_idx],
                     color="black",
                     fontsize=50, ha="center", va="bottom")  # Titles for non earth planets

        os.makedirs("figs", exist_ok=True)
        fig_dir = "figs"
        out_path = os.path.join(fig_dir, f"force_{PLANET_NAMES[planet_idx].lower()}.pdf")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)




def add_arrow(ax, x, y, u, v, scale, color, alpha=1.0):
    """Plot a single 3D arrow at z=0, with per-arrow alpha."""
    return ax.quiver(
        [x], [y], [0],
        [u*scale], [v*scale], [0],
        length=1, normalize=False,
        color=color,
        linewidth=2.5, arrow_length_ratio=0.1,
        alpha=alpha
    )


def make_force_animation(all_preds, all_truth, all_obs, planet_idx=2):
    # Reset rcparams to default
    plt.rcParams.update(plt.rcParamsDefault)
    
    force_scale = 18
    save_dir = "figs"
    gif_name    = f"forces_{PLANET_NAMES[planet_idx].lower()}.gif"

    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor":    "#111111",
        "figure.facecolor":  "black",
        "grid.color":        "#444444",
        "grid.alpha":        0.4,
        "legend.framealpha": 0.8,
        "text.color":        "white",
        "xtick.color":       "white",
        "ytick.color":       "white",
    })

    force_true  = all_truth[planet_idx]
    force_pred  = all_preds[planet_idx]
    obs         = all_obs[planet_idx]

    # Only include the first 100 timesteps.  
    timesteps = np.arange(0, len(obs), 2) #  35 for prototyping. 
    timesteps = timesteps[:100]

    # rescale so all orbits fit nicely
    smas        = [0.387,0.723,1.000,1.524,5.203,9.537,19.189,30.069]
    scale       = smas[-1] / smas[planet_idx]
    obs         = scale * obs

    # axis limits
    xmin, xmax = obs[:, 0].min() + 5, obs[:, 0].max() - 5
    ymin, ymax = obs[:, 1].min() + 5, obs[:, 1].max() - 5
    zmin, zmax = -25, 25  # same for all

    # ——— Set up figure & static elements ———
    fig = plt.figure(figsize=(14,8), facecolor="black")
    ax_t = fig.add_subplot(1,2,1, projection="3d", facecolor="black")
    ax_p = fig.add_subplot(1,2,2, projection="3d", facecolor="black")

    for ax in (ax_t, ax_p):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.view_init(elev=30, azim=45)
        ax.scatter(0,0,0, s=7000,  
                   c="gold", edgecolors="darkorange", zorder=3,
                   alpha=0.6)
        ax.tick_params(colors="white")

        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor("#333333")
            pane.set_facecolor("#111111")
            pane.set_alpha(1.00)

    ax_t.set_title(f"True",      fontsize=34, y=1.05)
    ax_p.set_title(f"Transformer", fontsize=34, y=1.05)

    true_color = "#00FFFF" # cyan
    pred_color = "#FF1493" # magenta
    true_handle = Line2D([0],[0], color=true_color, lw=2, label="True force")
    pred_handle = Line2D([0],[0], color=pred_color, lw=2, label="Predicted force")

    # proxy for the planet
    earth_handle = Line2D(
        [], [], 
        marker='o', linestyle='None', 
        markersize=10,
        markerfacecolor=PLANET_COLORS[planet_idx],
        label=PLANET_NAMES[planet_idx]
    )

    # proxy for the sun
    sun_handle = Line2D(
        [], [], 
        marker='o', linestyle='None', 
        markersize=12,
        markeredgecolor='darkorange',
        markerfacecolor='gold',
        label='Sun'
    )

    ax_t.legend(handles=[earth_handle, sun_handle, true_handle,],
                loc="upper center", ncols=3, fontsize=22,
                handlelength=1.2, handletextpad=0.4, columnspacing=0.8,
                bbox_to_anchor=(0.5, 0.93))
    ax_p.legend(handles=[earth_handle, sun_handle, pred_handle],
                loc="upper center", ncols=3, fontsize=22,
                handlelength=1.2, handletextpad=0.4, columnspacing=0.8,
                bbox_to_anchor=(0.5, 0.93))


    dynamic_artists = []
    HISTORY = 25  # fade length in frames 
    def update(i):
        # clear old artists
        for art in dynamic_artists:
            art.remove()
        dynamic_artists.clear()

        # only keep the last HISTORY frames
        start_k = max(0, i - HISTORY)
        for k in range(start_k, i + 1):
            t = timesteps[k]
            x, y = obs[t]

            # age = how many frames ago this obs was drawn
            age = i - k
            max_planet_alpha = 0.3
            force_alpha = 1.0 - (age / HISTORY)
            planet_alpha = max_planet_alpha - (age / HISTORY) * max_planet_alpha
            # clamp just in case
            force_alpha = max(0.0, min(1.0, force_alpha))
            planet_alpha = max(0.0, min(max_planet_alpha, planet_alpha))

            # RGBA color for quiver
            true_rgba = mcolors.to_rgba(true_color, force_alpha)
            pred_rgba = mcolors.to_rgba(pred_color, force_alpha)

            # draw true force
            a_t = ax_t.quiver(
                [x], [y], [0],
                [force_true[t,0]*force_scale],
                [force_true[t,1]*force_scale],
                [0], length=1, normalize=False,
                color=[true_rgba], linewidth=3., 
                arrow_length_ratio=0.2 
            )
            s_t = ax_t.scatter(
                x, y, 0,
                c=[PLANET_COLORS[planet_idx]],
                s=300,
                alpha=planet_alpha,
                zorder=5
            )

            # draw predicted force
            a_p = ax_p.quiver(
                [x], [y], [0],
                [force_pred[t,0]*force_scale],
                [force_pred[t,1]*force_scale],
                [0], length=1, normalize=False,
                color=[pred_rgba], linewidth=3., 
                arrow_length_ratio=0.2
            )
            s_p = ax_p.scatter(
                x, y, 0,
                c=[PLANET_COLORS[planet_idx]],
                s=300, 
                alpha=planet_alpha,
                zorder=5
            )

            dynamic_artists.extend([a_t, s_t, a_p, s_p])

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1,  wspace=0.02 )

    # ——— Build FuncAnimation ———
    anim = FuncAnimation(fig, update,
                         frames=len(timesteps),
                         interval=500,  # ms between frames
                         blit=False)

    # ——— Save GIF to disk ———
    anim.save(os.path.join(save_dir, gif_name), writer=PillowWriter(fps=12), dpi=75) 
    print(f"Saved GIF at {os.path.join(save_dir, gif_name)}")


def main():
    checkpoint = "next_token_pt_force_vector_transfer"
    model = load_model(checkpoint)
    model.eval()

    with (PHYSICS_CONFIG_DIR / ("force_vector_config.yaml")).open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    obs = np.load(PHYSICS_DATA_DIR / "obs_solar_system_two_body.npy")
    true_state = np.load(PHYSICS_DATA_DIR / "state_solar_system_two_body.npy")
    true_force_vectors = np.load(PHYSICS_DATA_DIR / "force_vector_solar_system_two_body.npy")

    all_preds = []
    all_truth = []
    all_obs = []
    for planet_idx in range(obs.shape[0]):
        x = torch.from_numpy(obs[planet_idx]).long().unsqueeze(0)[:, :-1, :]
        with torch.no_grad():
            all_logits = model(x)
            x_real, y_real = all_logits[0, :, 0], all_logits[0, :, 1]

        preds = np.stack([x_real.cpu().numpy(), y_real.cpu().numpy()], axis=-1)
        truth = true_force_vectors[planet_idx][:-1, :2]
        initial_pos = true_state[planet_idx, 1, 0:2]
        # Remove sun indices. 
        preds = preds[1::2, :]
        truth = truth[1::2, :]
        planet_obs = undiscretize_data(obs[planet_idx, :-1, :][1::2, :], config["input_vocab_size"] - 3, -50, 50)
        all_preds.append(preds)
        all_truth.append(truth)
        all_obs.append(planet_obs)

    # First make the pictures used in Figure 1. 
    make_force_pictures(all_preds, all_truth, all_obs)
    # Also make the force animation. 
    make_force_animation(all_preds, all_truth, all_obs)


if __name__ == "__main__":
    main()