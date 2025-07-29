import argparse
from functools import partial
from typing import Callable, Optional
import logging
import multiprocessing as mp
from pathlib import Path
import yaml

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import tqdm

from inductivebiasprobes import (
    build_exoplanet_distributions,
    build_two_body_distributions,
    generate_solar_system,
    generate_trajectories,
    sample_exoplanets,
)
from inductivebiasprobes.paths import PHYSICS_CONFIG_DIR, PHYSICS_DATA_DIR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Solar-system constants (planet masses are in units of solar masses).
# Order: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
SOLAR_SYSTEM_MASSES = [
    1.652e-07,  # Mercury
    2.447e-06,  # Venus
    3.003e-06,  # Earth
    3.213e-07,  # Mars
    9.544e-04,  # Jupiter
    2.857e-04,  # Saturn
    4.354e-05,  # Uranus
    5.165e-05,  # Neptune
]

SUN_MASS = 1.0

# Semi-major axes (AU)
SOLAR_SYSTEM_SMAS = [
    0.387,
    0.723,
    1.000,
    1.524,
    5.203,
    9.537,
    19.189,
    30.069,
]

# Orbital eccentricities
SOLAR_SYSTEM_ECCS = [
    0.205,
    0.007,
    0.017,
    0.093,
    0.048,
    0.054,
    0.047,
    0.009,
]

SMA_MAX = 32.0 
TRAJ_XY_MIN = -50.0
TRAJ_XY_MAX = 50.0
FORCE_MAG_MIN = 2e-7
FORCE_MAG_MAX = 5e-3

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate data for the two-body problem (parallel)."
    )
    parser.add_argument(
        "--num_points_per_trajectory",
        type=int,
        default=1_000,
        help="Number of points to generate along the orbit.",
    )
    parser.add_argument(
        "--num_train_trajectories",
        type=int,
        # default=1_000,         # 1K for prototyping. 
        default=10_000_000,  # 10M is for the full pretraining. 
        help="Number of trajectories to generate (train).",
    )
    parser.add_argument(
        "--num_val_trajectories",
        type=int,
        default=300,
        help="Number of trajectories to generate (val).",
    )
    parser.add_argument(
        "--num_test_trajectories",
        type=int,
        default=300,
        help="Number of trajectories to generate (test).",
    )
    parser.add_argument(
        "--dts",
        type=float,
        nargs="+",
        help="Time step in years",
        default=[7 * 0.0027, 6 * 30 * 0.0027],  # 1 week, 6 months
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=7_000,
        help="Number of bins for discretizing the data (-1 for no discretization).",
    )
    parser.add_argument(
        "--total_force_magnitudes",
        type=int,
        default=10_000,
        help="Total number of force magnitudes to generate (train and test combined).",
    )
    parser.add_argument(
        "--num_unmasked_force_magnitudes",
        type=int,
        default=9000,
        help="""
        Number of force magnitudes to completely unmask for training (the rest 
        will have two unmasked per sequence and the remainder of the sequence 
        will be used for test).
        """,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=90,
        help="Number of parallel workers to use in multiprocessing pool",
    )
    parser.add_argument(
        "--two_body_only",
        action="store_true",
        help="Only generate two-body problems",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode without parallel workers",
    )
    return parser.parse_args()


def discretize_trajectory(traj, num_bins, min_value, max_value):
    """Discretize traj into indices 0 ... num_bins-1."""
    # keep values strictly inside the interval
    clipped = np.clip(traj, min_value + 1e-12, max_value - 1e-12)
    lo, hi = min_value, max_value
    width = (hi - lo) / num_bins
    idx = np.floor((clipped - lo) / width).astype(int)
    bins = np.clip(idx, 0, num_bins - 1)
    return bins


def process_multiplanet_trajectory(
    obs_list,
    heavier_obs,
    num_points,
    num_bins,
    min_value=TRAJ_XY_MIN,
    max_value=TRAJ_XY_MAX,
    dt_token=0,
):
    planets_traj = np.array([obs.trajectory for obs in obs_list])
    sun_traj = np.array([heavier_obs.trajectory])
    planets_and_sun_traj_3d = np.concatenate([planets_traj, sun_traj], axis=0)
    # Flatten to 2D making sure to interleave the planets' trajectories.
    traj = planets_and_sun_traj_3d.transpose(1, 0, 2).reshape(-1, 2)#[:num_points, :]
    traj_raw = traj
    if num_bins > 0:
        traj = discretize_trajectory(traj, num_bins, min_value, max_value)
    # Add dt_token as first timestep.
    dt_token_timestep = np.full((1, 2), dt_token)
    traj = np.concatenate([dt_token_timestep, traj], axis=0)
    traj = traj[:num_points, :]
    return traj_raw, traj


def process_multiplanet_states(
    state_list,
    heavier_obs,
    num_points,
):
    """Process *state* information into a stacked representation matching the
    layout used for observations.
    """
    planets_traj = np.array([state.trajectory_light for state in state_list])
    sun_traj = np.tile(heavier_obs.trajectory[0], (num_points, 1))[np.newaxis]
    planets_and_sun_traj_3d = np.concatenate(
        [planets_traj, sun_traj[:, : len(planets_traj[0])]], axis=0
    )
    planets_and_sun_traj_3d_flat = planets_and_sun_traj_3d.transpose(1, 0, 2).reshape(
        -1, 2
    )[:num_points, :]
    relative_position = planets_and_sun_traj_3d_flat - sun_traj[0]
    sun_traj = sun_traj[0]
    relative_velocity = np.array([state.relative_velocity for state in state_list])
    sun_relative_velocity = np.zeros((1, len(relative_velocity[0]), 2))
    planets_and_sun_relative_velocity_3d = np.concatenate(
        [relative_velocity, sun_relative_velocity], axis=0
    )
    relative_velocity = planets_and_sun_relative_velocity_3d.transpose(1, 0, 2).reshape(
        -1, 2
    )[:num_points, :]

    # The states for the sun positions aren't used so we arbitrarily set to 0. 
    planet_masses = np.array([state.m_light for state in state_list] + [0.0])
    planet_masses_2d = np.repeat(planet_masses[np.newaxis, :], num_points, axis=0).T
    planet_masses = planet_masses_2d.transpose(1, 0).reshape(-1)[:num_points][
        :, np.newaxis
    ]
    sun_mass = np.tile(state_list[0].m_heavy, (num_points, 1))
    relative_masses = planet_masses / sun_mass

    # Add 0s as first timestep to account for dt token
    relative_position = np.concatenate([np.zeros((1, 2)), relative_position], axis=0)[:num_points, :]
    relative_velocity = np.concatenate([np.zeros((1, 2)), relative_velocity], axis=0)[:num_points, :]
    relative_masses = np.concatenate([np.zeros((1, 1)), relative_masses], axis=0)[:num_points, :]
    planet_masses = np.concatenate([np.zeros((1, 1)), planet_masses], axis=0)[:num_points, :]
    sun_mass = np.concatenate([np.zeros((1, 1)), sun_mass], axis=0)[:num_points, :]

    return relative_position, relative_velocity, relative_masses, planet_masses, sun_mass


def process_multiplanet_fn_of_state(fn_of_state_list, num_points, force_magnitude_mask_id, force_vector_mask_id):
    planet_force_vectors = np.array([x.force_vector for x in fn_of_state_list])
    force_vector_dim = planet_force_vectors.shape[-1]
    sun_force_vector = np.sum(planet_force_vectors, axis=0, keepdims=True)
    force_vectors_3d = np.concatenate([planet_force_vectors, sun_force_vector], axis=0)
    force_vectors = force_vectors_3d.transpose(1, 0, 2).reshape(-1, force_vector_dim)[
        :num_points, :
    ]
    planet_force_magnitudes = np.linalg.norm(planet_force_vectors, axis=-1)
    sun_force_magnitude = np.linalg.norm(sun_force_vector, axis=-1)
    force_magnitudes_2d = np.concatenate(
        [planet_force_magnitudes, sun_force_magnitude], axis=0
    )
    force_magnitudes = force_magnitudes_2d.transpose(1, 0).reshape(-1, 1)[
        :num_points, :
    ]
    # Standardize force vectors to have unit interval (just for solar system). 
    max_force_vec = np.max(np.abs(force_vectors), axis=0)
    force_vectors = force_vectors / max_force_vec
    # Add 0s as first timestep to account for dt token
    force_vectors = np.concatenate([np.zeros((1, force_vector_dim)), force_vectors], axis=0)[:num_points, :]
    force_magnitudes = np.concatenate([np.zeros((1, 1)), force_magnitudes], axis=0)[:num_points, :]
    # Because states are only used for two-body problems we can mask the sun's force outputs. 
    # Set every even timestep to the mask index, both for force vectors and magnitudes
    even_timesteps = np.arange(0, num_points, 2)
    force_vectors[even_timesteps, :] = force_vector_mask_id
    force_magnitudes[even_timesteps, :] = force_magnitude_mask_id
    return force_vectors, force_magnitudes


def _generate_single_trajectory(task):
    """
    Worker function for a single trajectory.
    task is a tuple: (e, seed, num_points_per_trajectory, dt)
    """
    (
        eccs,
        smas,
        mass_1s,
        mass_2,
        seed,
        num_points,
        dts,
        dt_to_token,
        num_bins,
        force_magnitude_mask_id,
        force_vector_mask_id,
        skip_states,
    ) = task

    # Generate a random two-body problem
    while True:
        try:
            problem = generate_solar_system(
                eccs,
                smas,
                mass_1s,
                mass_2,
                seed=seed,
            )

            # Randomly generate a single dt for each trajectory. 
            random_state = np.random.RandomState(seed + 10)
            dt_ind = random_state.choice(len(dts), size=1, replace=True).item()
            dt = dts[dt_ind]
            dt_token = dt_to_token[dt_ind]
            obs_list, state_list, fn_of_state_list, heavier_obs = (
                generate_trajectories(
                    problem,
                    num_points // 2,
                    dt,
                    rng=seed + 10,  # or some offset
                    fix_heavier=len(mass_1s) > 1,
                )
            )
            break
        except Exception as e:
            logger.warning(f"Error generating trajectory: {e}")
            logger.warning(
                f"eccs: {eccs}, smas: {smas}, mass_1s: {mass_1s}, mass_2: {mass_2}"
            )
            seed += 1

    # Convert relevant data to arrays for easier concatenation later
    # Observations
    traj_raw, traj = process_multiplanet_trajectory(
        obs_list,
        heavier_obs,
        num_points,
        num_bins,
        dt_token=dt_token,
    )

    # States
    if not skip_states:
        relative_position, relative_velocity, relative_masses, planet_masses, sun_mass = (
            process_multiplanet_states(
                state_list,
                heavier_obs,
                num_points,
            )
        )
        full_state = np.concatenate(
            [relative_position, relative_velocity, planet_masses, sun_mass],
            axis=1,
        )
        state = np.concatenate(
            [relative_position, relative_velocity, relative_masses],
            axis=1,
        )
        # Functions of state
        force_vectors, force_magnitudes = process_multiplanet_fn_of_state(
            fn_of_state_list, num_points, force_magnitude_mask_id, force_vector_mask_id
        )
    else:
        state = None
        full_state = None
        force_vectors = None
        force_magnitudes = None
    return (traj_raw, traj, full_state, state, force_vectors, force_magnitudes)


def _sample_exoplanet_for_queue(
    seed_offset,
    exoplanet_distributions,
    num_points_per_trajectory,
    dts,
    dt_to_token,
    num_bins,
    label,
    force_magnitude_mask_id,
    force_vector_mask_id,
    skip_states,
):
    """Worker function for sampling exoplanets in parallel"""
    cur_seed = seed_offset
    eccs, smas, mass_1s, mass_2 = sample_exoplanets(
        exoplanet_distributions,
        num_planets=1 if "two_body" in label else None,
        seed=cur_seed,
    )
    return (
        eccs,
        smas,
        mass_1s,
        mass_2,
        cur_seed,
        num_points_per_trajectory,
        dts,
        dt_to_token,
        num_bins,
        force_magnitude_mask_id,
        force_vector_mask_id,
        skip_states,
    )


def generate_data_parallel(
    exoplanet_distributions: dict,
    num_points_per_trajectory: int,
    num_trajectories: int,
    dts: list[float],
    dt_to_token: dict[int, int],
    seed: int,
    num_workers: int,
    out_dir: Path,
    label: str,
    num_bins: int = -1,
    skip_states: bool = False,
    force_magnitude_mask_id: float = float("inf"),
    force_vector_mask_id: float = float("inf"),
    dtype: np.dtype = np.dtype(np.float32),
    debug: bool = False,
    sample_fn_override: Optional[Callable] = None,
) -> int:
    """
    Generate data and stream it straight to three .npy files that
    are memory-mapped on disk.  Nothing large stays in RAM.
    Returns the next seed to keep your RNG reproducible.
    """
    # -------- 1. Determine final shapes up front ----------
    n_traj = num_trajectories
    obs_dim = 2  #  (x, y)
    state_dim = 5  # 2*pos + 2*vel + relative_mass
    full_state_dim = 6  # 2*pos + 2*vel + m1 + m2
    force_vec_dim = 2
    force_mag_dim = 1

    obs_shape = (n_traj, num_points_per_trajectory, obs_dim)
    # Pretraining doesn't need states. 
    if not skip_states:
        state_shape = (n_traj, num_points_per_trajectory, state_dim)
        full_state_shape = (n_traj, num_points_per_trajectory, full_state_dim)
        force_vector_shape = (n_traj, num_points_per_trajectory, force_vec_dim)
        force_magnitude_shape = (n_traj, num_points_per_trajectory, force_mag_dim)
        force_magnitude_dtype = np.float32
    if num_bins > 0:
        dtype = np.dtype(np.uint16)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # -------- 2. Create *empty* .npy files as memmaps -----
    obs_mm = open_memmap(
        out_dir / f"obs_{label}.npy",
        mode="w+",
        dtype=dtype,
        shape=obs_shape,
    )
    if not skip_states:
        state_mm = open_memmap(
            out_dir / f"state_{label}.npy",
            mode="w+",
            dtype=np.float32,
            shape=state_shape,
        )
        full_state_mm = open_memmap(
            out_dir / f"full_state_{label}.npy",
            mode="w+",
            dtype=np.float32,
            shape=full_state_shape,
        )
        force_vector_mm = open_memmap(
            out_dir / f"force_vector_{label}.npy",
            mode="w+",
            dtype = np.float32, 
            shape=force_vector_shape,
        )
        force_magnitude_mm = open_memmap(
            out_dir / f"force_magnitude_{label}.npy",
            mode="w+",
            dtype=force_magnitude_dtype,
            shape=force_magnitude_shape,
        )

    # -------- 3. Build the work queue in parallel ---------------------
    seed_offsets = [
        seed + i * 20 for i in range(num_trajectories)
    ]  # reproducible RNG stream

    # Allow callers to override the sampling strategy (e.g. planet-specific uniform sampling).
    if sample_fn_override is not None:
        sample_fn = sample_fn_override
    else:
        sample_fn = partial(
            _sample_exoplanet_for_queue,
            exoplanet_distributions=exoplanet_distributions,
            num_points_per_trajectory=num_points_per_trajectory,
            dts=dts,
            dt_to_token=dt_to_token,
            num_bins=num_bins,
            label=label,
            force_magnitude_mask_id=force_magnitude_mask_id,
            force_vector_mask_id=force_vector_mask_id,
            skip_states=skip_states,
        )

    if debug:
        tasks = [
            sample_fn(s)
            for s in tqdm.tqdm(seed_offsets, desc=f"[{label}] Generating queue")
        ]
    else:
        with mp.Pool(processes=num_workers) as pool:
            tasks = list(
                tqdm.tqdm(
                    pool.imap(sample_fn, seed_offsets),
                    total=len(seed_offsets),
                    desc=f"[{label}] Generating queue",
                )
            )

    # Next seed for reproducibility
    cur_seed = seed + num_trajectories * 20

    # -------- 4. Worker pool + streamed writes ------------
    if debug:
        for idx, task in enumerate(tqdm.tqdm(tasks, desc=f"[{label}] Debug mode")):
            traj_raw, traj, full_state, state, force_vectors, force_magnitudes = (
                _generate_single_trajectory(task)
            )
            obs_mm[idx] = traj.astype(dtype, copy=False)
            if not skip_states:
                state_mm[idx] = state.astype(np.float32, copy=False)
                full_state_mm[idx] = full_state.astype(np.float32, copy=False)
                force_vector_mm[idx] = force_vectors.astype(np.float32, copy=False)
                force_magnitude_mm[idx] = force_magnitudes.astype(force_magnitude_dtype, copy=False)
    else:
        with mp.Pool(processes=num_workers) as pool:
            for idx, (
                traj_raw,
                traj,
                full_state,
                state,
                force_vectors,
                force_magnitudes,
            ) in enumerate(
                tqdm.tqdm(
                    pool.imap(_generate_single_trajectory, tasks),
                    total=len(tasks),
                    desc=f"[{label}]",
                )
            ):
                #  write the current trajectory directly to disk
                obs_mm[idx] = traj.astype(dtype, copy=False)
                if not skip_states:
                    state_mm[idx] = state.astype(np.float32, copy=False)
                    full_state_mm[idx] = full_state.astype(np.float32, copy=False)
                    force_vector_mm[idx] = force_vectors.astype(np.float32, copy=False)
                    force_magnitude_mm[idx] = force_magnitudes.astype(
                        force_magnitude_dtype, copy=False
                    )
    obs_mm.flush()
    del obs_mm
    if not skip_states:
        state_mm.flush()
        del state_mm
        full_state_mm.flush()
        del full_state_mm
        force_vector_mm.flush()
        del force_vector_mm
        force_magnitude_mm.flush()
        del force_magnitude_mm

    return cur_seed  # so main() can keep advancing the seed


def main():
    args = parse_args()
    logger.info(f"Generating data with args: {args}")

    PHYSICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PHYSICS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    dt_to_token = {0: args.num_bins, 1: args.num_bins + 1}
    force_vector_mask_id = float("inf")
    force_magnitude_mask_id = float("inf")
    
    # =============================================================
    # Generate solar system as single trajectory (used for plotting orbits).
    # =============================================================
    solar_system_problem = generate_solar_system(
        eccs=SOLAR_SYSTEM_ECCS,
        smas=SOLAR_SYSTEM_SMAS,
        mass_1s=SOLAR_SYSTEM_MASSES,
        mass_2=SUN_MASS,
        seed=seed,
        center_of_mass=np.array([0, 0]),
    )
    obs_list, _, _, heavier_obs = (
        generate_trajectories(
            solar_system_problem,
            args.num_points_per_trajectory,
            dt=args.dts[0],
            rng=0,  # or some offset
        )
    )
    traj_raw, traj = process_multiplanet_trajectory(
        obs_list,
        heavier_obs,
        args.num_points_per_trajectory,
        args.num_bins,
        dt_token=dt_to_token[0],
    )
    np.save(
        PHYSICS_DATA_DIR / "obs_solar_system_single_sequence.npy", traj,
    )
    
    
    # =============================================================
    # Generate solar system as 8 two-body problems (for forces)
    # =============================================================

    num_planets = len(SOLAR_SYSTEM_MASSES)
    solar_system_two_body_obs = []
    solar_system_two_body_state = []
    solar_system_two_body_full_state = []
    solar_system_two_body_force_vectors = []
    solar_system_two_body_force_magnitudes = []
    traj_raws = []
    for i in range(num_planets):
        curr_problem = generate_solar_system(
            eccs=[SOLAR_SYSTEM_ECCS[i]],
            smas=[SOLAR_SYSTEM_SMAS[i]],
            mass_1s=[SOLAR_SYSTEM_MASSES[i]],
            mass_2=SUN_MASS,
            seed=seed,
            center_of_mass=np.array([0, 0]),
        )
        obs_list, state_list, fn_of_state_list, heavier_obs = (
            generate_trajectories(
                curr_problem,
                args.num_points_per_trajectory,
                dt=args.dts[1], # Do the longer dt so it covers the full orbit. 
                rng=0,
                fix_heavier=False,
            )
        )
        traj_raw, traj = process_multiplanet_trajectory(
            obs_list,
            heavier_obs,
            args.num_points_per_trajectory,
            args.num_bins,
            dt_token=dt_to_token[1], 
        )
        relative_position, relative_velocity, relative_masses, planet_masses, sun_mass = (
            process_multiplanet_states(
                state_list,
                heavier_obs,
                args.num_points_per_trajectory,
            )
        )
        full_state = np.concatenate(
            [relative_position, relative_velocity, planet_masses, sun_mass],
            axis=1,
        )
        state = np.concatenate(
            [relative_position, relative_velocity, relative_masses],
            axis=1,
        )
        force_vectors, force_magnitudes = process_multiplanet_fn_of_state(
            fn_of_state_list, args.num_points_per_trajectory,
            force_magnitude_mask_id, force_vector_mask_id,
        )
        traj_raws.append(traj_raw)
        solar_system_two_body_obs.append(traj)
        solar_system_two_body_state.append(state)
        solar_system_two_body_full_state.append(full_state)
        solar_system_two_body_force_vectors.append(force_vectors)
        solar_system_two_body_force_magnitudes.append(force_magnitudes)
    traj_raws = np.stack(traj_raws, axis=0)
    logger.info(
        f"solar system two body: traj_x min: {traj_raws[:, :, 0].min()}, traj_x max: {traj_raws[:, :, 0].max()}"
    )
    logger.info(
        f"solar system two body: traj_y min: {traj_raws[:, :, 1].min()}, traj_y max: {traj_raws[:, :, 1].max()}"
    )
    solar_system_two_body_obs = np.stack(solar_system_two_body_obs, axis=0)
    solar_system_two_body_state = np.stack(solar_system_two_body_state, axis=0)
    solar_system_two_body_full_state = np.stack(solar_system_two_body_full_state, axis=0)
    solar_system_two_body_force_vectors = np.stack(solar_system_two_body_force_vectors, axis=0)
    solar_system_two_body_force_magnitudes = np.stack(solar_system_two_body_force_magnitudes, axis=0)
    np.save(
        PHYSICS_DATA_DIR / "obs_solar_system_two_body.npy", solar_system_two_body_obs
    )
    np.save(
        PHYSICS_DATA_DIR / "state_solar_system_two_body.npy",
        solar_system_two_body_state,
    )
    np.save(
        PHYSICS_DATA_DIR / "full_state_solar_system_two_body.npy",
        solar_system_two_body_full_state,
    )
    np.save(
        PHYSICS_DATA_DIR / "force_vector_solar_system_two_body.npy",
        solar_system_two_body_force_vectors,
    )
    np.save(
        PHYSICS_DATA_DIR / "force_magnitude_solar_system_two_body.npy",
        solar_system_two_body_force_magnitudes,
    )

    # When we train on the forces, we mask most of the force vectors and perform
    # inference on the unmasked ones at test time. 
    solar_system_two_body_force_vectors_masked = solar_system_two_body_force_vectors.copy()
    unmasked_per_sequence = 10
    rs_mask = np.random.RandomState(0) 
    for i in range(solar_system_two_body_force_vectors.shape[0]):
        # Unmask odd indices because even indices correspond to the sun positions. 
        odd_indices = np.arange(1, solar_system_two_body_force_vectors.shape[1], 2)
        unmasked_sample_inds = rs_mask.choice(odd_indices, size=unmasked_per_sequence, replace=False)
        masked_sample_inds = np.setdiff1d(np.arange(solar_system_two_body_force_vectors.shape[1]), unmasked_sample_inds)
        solar_system_two_body_force_vectors_masked[i, masked_sample_inds] = force_vector_mask_id
    np.save(
        PHYSICS_DATA_DIR / "force_vector_solar_system_two_body_masked.npy",
        solar_system_two_body_force_vectors_masked,
    )

    # For other solar systems, load data of known exoplanets from the NASA Exoplanet Archive. 
    exoplanet_df = pd.read_csv("exoplanet_data.csv")
    exoplanet_distributions = build_exoplanet_distributions(
        exoplanet_df, max_sma=SMA_MAX
    )
    if args.two_body_only:
        exoplanet_distributions = build_two_body_distributions(max_sma=SMA_MAX)
    for n_traj, label in zip(
        (
            args.num_train_trajectories,
            args.num_val_trajectories,
            args.num_test_trajectories,
            1_000,
            args.total_force_magnitudes,
            300, # for force val/test
            300,
        ),
        (
            "train",
            "val",
            "test",
            "traj",
            "two_body_train",
            "two_body_val",
            "two_body_test",
        ),
    ):
        seed = generate_data_parallel(
            exoplanet_distributions=exoplanet_distributions,
            num_points_per_trajectory=args.num_points_per_trajectory,
            num_trajectories=n_traj,
            dts=args.dts,
            dt_to_token=dt_to_token,
            seed=seed,
            num_workers=args.num_workers,
            out_dir=PHYSICS_DATA_DIR,
            label=label,
            num_bins=args.num_bins,
            dtype=np.float16,  # or np.float16 if you like
            skip_states='two_body' not in label, # skip states for pretraining. 
            force_magnitude_mask_id=force_magnitude_mask_id,
            force_vector_mask_id=force_vector_mask_id,
            debug=args.debug,
        )
    
    # The force magnitudes use multiple solar systems (so m1 varies). 
    # Unmask some of the force magnitudes for training. 
    force_magnitudes_masked = np.load(PHYSICS_DATA_DIR / "force_magnitude_two_body_train.npy")
    rs_mask = np.random.RandomState(0)
    # For the first args.num_unmasked_force_magnitudes sequences, all outputs
    # will be unmasked for the model to train on. 
    # For the rest, unmask 2 samples per test sequence. 
    for i in range(args.num_unmasked_force_magnitudes, force_magnitudes_masked.shape[0]):
        odd_indices = np.arange(1, force_magnitudes_masked.shape[1]-1, 2)
        unmasked_sample_inds = rs_mask.choice(odd_indices, size=2, replace=False)
        masked_sample_inds = np.setdiff1d(np.arange(force_magnitudes_masked.shape[1]), unmasked_sample_inds)
        force_magnitudes_masked[i, masked_sample_inds] = force_magnitude_mask_id
    np.save(PHYSICS_DATA_DIR / "force_magnitude_two_body_train_masked.npy", force_magnitudes_masked)

    # Save configs
    common_config = {
        "input_dim": 2,
        "block_size": args.num_points_per_trajectory - 1,
    }

    ntp_mask_id = args.num_bins if args.num_bins != -1 else float("inf")
    ntp_config = {
        **common_config,
        "num_data_points": args.num_train_trajectories,
        "output_dim": 2,
        "predict_type": "next_token",
        "mask_id": ntp_mask_id,
    }
    if args.num_bins != -1:
        ntp_config["output_vocab_size"] = args.num_bins + 1 + len(args.dts) # +1 for pad token
        ntp_config["input_vocab_size"] = args.num_bins + 1 + len(args.dts)
    with open(PHYSICS_CONFIG_DIR / "ntp_config.yaml", "w") as f:
        yaml.dump(ntp_config, f)

    for transfer_name in (
        "force_vector",
        "force_magnitude",
    ):
        config = {
            **common_config,
            "num_data_points": args.total_force_magnitudes if transfer_name == "force_magnitude" else len(SOLAR_SYSTEM_MASSES),
            "output_dim": (2 if transfer_name == "force_vector" else 1),
            "predict_type": transfer_name,
            "force_min": FORCE_MAG_MIN if transfer_name == "force_magnitude" else None,
            "force_max": FORCE_MAG_MAX if transfer_name == "force_magnitude" else None,
            "mask_id": force_vector_mask_id if transfer_name == "force_vector" else force_magnitude_mask_id
        }
        if args.num_bins != -1:
            config["input_vocab_size"] = args.num_bins + 1 + len(args.dts)
        config["output_vocab_size"] = None
        with open(PHYSICS_CONFIG_DIR / f"{transfer_name}_config.yaml", "w") as f:
            yaml.dump(config, f)



if __name__ == "__main__":
    main()
