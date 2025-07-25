from typing import Optional

import numpy as np
from numpy.random import Generator
from pydantic import BaseModel, Field
from scipy.optimize import newton
from scipy import stats

float_type = float | np.floating


class TwoBodyProblem(BaseModel):
    model_config: dict = {
        "arbitrary_types_allowed": True,
    }

    mass_1: float_type
    mass_2: float_type
    r_0_magnitude: float_type = Field(
        ..., description="Magnitude of the initial relative position vector"
    )
    r_0_angle: float_type = Field(
        ...,
        description="Angle in the x-y plane of the initial relative position vector",
    )
    v_0_magnitude: float_type = Field(
        ..., description="Magnitude of the initial relative velocity vector"
    )
    v_0_angle: float_type = Field(
        ...,
        description="Angle in the x-y plane of the initial relative velocity vector",
    )
    center_of_mass: tuple[float_type, float_type] = Field(
        (0, 0), description="Center of mass of the two objects"
    )
    gravitational_constant: float_type = 39.4784176044  # AU^3/(year^2 * solar_mass)


class MultipleTwoBodyProblem(BaseModel):
    model_config: dict = {
        "arbitrary_types_allowed": True,
    }

    mass_1s: list[float_type]
    mass_2: float_type
    r_0_magnitudes: list[float_type] = Field(
        ..., description="Magnitude of the initial relative position vector"
    )
    r_0_angles: list[float_type] = Field(
        ...,
        description="Angle in the x-y plane of the initial relative position vector",
    )
    v_0_magnitudes: list[float_type] = Field(
        ..., description="Magnitude of the initial relative velocity vector"
    )
    v_0_angles: list[float_type] = Field(
        ...,
        description="Angle in the x-y plane of the initial relative velocity vector",
    )
    center_of_mass: tuple[float_type, float_type] = Field(
        (0, 0), description="Center of mass of the two objects"
    )
    gravitational_constant: float_type = 39.4784176044  # AU^3/(year^2 * solar_mass)


class TrajectoryObservation(BaseModel):
    trajectory: list


class TrajectoryState(BaseModel):
    model_config: dict = {
        "arbitrary_types_allowed": True,
    }

    trajectory_light: list
    trajectory_heavy: list
    relative_velocity: list
    m_light: float_type
    m_heavy: float_type
    gravitational_constant: float_type
    eccentricity: float_type  # Not actually used for Newton but helpful to carry around. 


class TrajectoryFunctionsOfState(BaseModel):
    force_vector: list


def build_exoplanet_distributions(df, max_sma: float = 32) -> dict:
    """Build a dictionary of distributions for the exoplanet parameters.

    Args:
        df: DataFrame containing the exoplanet data.
        max_sma: Maximum semi-major axis to consider.
        two_body_only: Whether to only consider two-body problems.

    Returns:
        A dictionary of distributions for the exoplanet parameters.
    """
    # ----------- 1. Eccentricity ------------
    # Use Beta distribution values from https://arxiv.org/pdf/1306.4982
    ecc_distribution = stats.beta(0.867, 3.03)

    # ----------- 2. Semi-major axis ------------
    sma_distribution = stats.uniform(0.3, max_sma - 0.3)

    # ----------- 3. Planet mass ------------
    solar_masses = df["planet_solar_mass"]
    solar_masses = solar_masses[solar_masses < np.percentile(solar_masses, 90)]
    shape, loc, scale = stats.gamma.fit(solar_masses)
    planet_mass_distribution = stats.gamma(shape, loc=loc, scale=scale)


    # ----------- 4. Star mass ------------
    star_masses = df["star_solar_mass"]
    shape, loc, scale = stats.lognorm.fit(star_masses)
    star_mass_distribution = stats.lognorm(shape, loc=loc, scale=scale)

    # ----------- 5. Number of planets ------------
    num_planets_distribution = stats.randint(1, 11)

    distributions = {
        "ecc": ecc_distribution,
        "sma": sma_distribution,
        "planet_mass": planet_mass_distribution,
        "star_mass": star_mass_distribution,
        "num_planets": num_planets_distribution,
    }

    return distributions


def build_two_body_distributions(max_sma: float = 32) -> dict:
    """Build a dictionary of distributions for the two-body problem parameters.

    Args:
        max_sma: Maximum semi-major axis to consider.

    Returns:
        A dictionary of distributions for the two-body problem parameters.
    """
    # ----------- 1. Eccentricity ------------
    ecc_distribution = stats.beta(0.867, 3.03)

    # ----------- 2. Semi-major axis ------------
    sma_distribution = stats.uniform(0.3, max_sma - 0.3)

    # ----------- 3. Masses ------------
    mass_distribution = stats.uniform(1e-4, 1e-2)

    distributions = {
        "ecc": ecc_distribution,
        "sma": sma_distribution,
        "planet_mass": mass_distribution,
        "star_mass": mass_distribution,
        "num_planets": stats.randint(1, 2),
        "two_body": True,
    }

    return distributions


def sample_exoplanets(
    exoplanet_distributions: dict, num_planets: int | None = None, seed: int = 0
) -> tuple[list[float_type], list[float_type], list[float_type], float_type]:
    """Sample exoplanets from the distributions.

    Args:
        exoplanet_distributions: Dictionary of distributions for the exoplanet parameters.
        seed: Random number generator seed.

    Returns:
        A tuple of lists of eccentricities, semi-major axes, planet masses, and star mass.
    """
    if num_planets is None:
        num_planets = (
            exoplanet_distributions["num_planets"].rvs(size=1, random_state=seed).item()
        )
        seed += 1
    
    if "two_body" in exoplanet_distributions:
        masses = exoplanet_distributions["star_mass"].rvs(size=2, random_state=seed)
        mass_1, mass_2 = np.min(masses), np.max(masses)
        seed += 1
        mass_1s = [mass_1]
    else:
        mass_2 = exoplanet_distributions["star_mass"].rvs(size=1, random_state=seed).item()
        seed += 1
        mass_2 = np.clip(mass_2, 0.1, 5.0).item()
        mass_1s = []

    smas = []
    eccs = []
    for i in range(num_planets):
        sma = exoplanet_distributions["sma"].rvs(size=1, random_state=seed).item()
        seed += 1
        ecc = exoplanet_distributions["ecc"].rvs(size=1, random_state=seed).item()
        seed += 1
        if "two_body" not in exoplanet_distributions:
            mass_1 = (
                exoplanet_distributions["planet_mass"].rvs(size=1, random_state=seed).item()
            )
            seed += 1
            mass_1 = np.clip(mass_1, 1e-9, 1e-3).item()
            mass_1s.append(mass_1)
        smas.append(sma)
        eccs.append(ecc)
        seed += 1

    return eccs, smas, mass_1s, mass_2


def generate_vector(magnitude: float_type, angle: float_type):
    """Generate a 3D vector from a magnitude and angle in the x-y plane.

    Args:
        magnitude: Magnitude of the vector.
        angle: Angle of the vector (in x-y plane).

    Returns:
        Numpy array representing the 3D vector.
    """
    return np.array([magnitude * np.cos(angle), magnitude * np.sin(angle), 0])


def calculate_orbital_parameters(
    pos_rel: np.ndarray, vel_rel: np.ndarray, mu: float_type
) -> tuple[float_type, float_type]:
    """
    Calculate orbital parameters (semi-major axis/parabola parameter and eccentricity).

    Args:
        pos_rel: Relative position vector
        vel_rel: Relative velocity vector
        mu: Standard gravitational parameter (G * (m1 + m2))

    Returns:
        Tuple of semi-major axis and eccentricity of the orbit.
    """
    # Specific orbital energy
    energy = np.linalg.norm(vel_rel) ** 2 / 2 - mu / np.linalg.norm(pos_rel)

    # Specific angular momentum vector
    ang_mom_specific = np.cross(pos_rel, vel_rel)
    h = np.linalg.norm(ang_mom_specific)

    if np.isclose(energy, 0, atol=1e-12):
        raise ValueError("Parabolic orbits are not supported.")

    sma = -mu / (2 * energy)
    ecc = np.sqrt(np.clip(1 + 2 * energy * h**2 / mu**2, 0, None))

    if ecc > 1:
        raise ValueError("Hyperbolic orbits are not supported.")

    return sma, ecc


def solve_kepler_equation(M: float_type, e: float_type) -> float_type:
    """Solve Kepler's Equation for elliptic orbits."""
    return newton(lambda E: E - e * np.sin(E) - M, M)


def true_anomaly_from_anomaly(anomaly: float_type, e: float_type) -> float_type:
    """Convert eccentric/hyperbolic anomaly to true anomaly."""
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(anomaly / 2))


def compute_position(nu: float_type, sma: float_type, ecc: float_type) -> np.ndarray:
    """Compute the position vector in the orbital plane.
    Assumes the entire orbit is in the x-y plane, before rotation to 3D.

    Args:
        nu: True anomaly
        sma: Semi-major axis
        ecc: Eccentricity

    Returns:
        Position vector in the orbital plane.
    """
    r = sma * (1 - ecc**2) / (1 + ecc * np.cos(nu))
    return np.array([r * np.cos(nu), r * np.sin(nu), 0])


def compute_velocity(
    nu: float_type, sma: float_type, ecc: float_type, mu: float_type
) -> np.ndarray:
    """Compute the velocity vector in the orbital plane.
    Assumes the entire orbit is in the x-y plane, before rotation to 3D.

    Args:
        nu: True anomaly
        sma: Semi-major axis
        ecc: Eccentricity
        mu: Standard gravitational parameter (G*(M+m))

    Returns:
        Velocity vector in the orbital plane.
    """
    v_r = np.sqrt(mu / (sma * (1 - ecc**2))) * ecc * np.sin(nu)
    v_theta = np.sqrt(mu / (sma * (1 - ecc**2))) * (1 + ecc * np.cos(nu))
    return np.array(
        [
            -v_r * np.sin(nu) + v_theta * np.cos(nu),
            v_r * np.cos(nu) + v_theta * np.sin(nu),
            0,
        ]
    )


def compute_orbit_in_new_frame(
    heavier_orbit: np.ndarray, lighter_orbit: np.ndarray, heavier_coord: np.ndarray
) -> np.ndarray:
    """Compute the lighter object's in a new reference frame where the heavier
    object is fixed at heavier_coord.

    Args:
        heavier_orbit: The original orbit of the heavier object
        lighter_orbit: The original orbit of the lighter object
        heavier_coord: The fixed coordinates of the heavier object

    Returns:
        The adjusted orbit of the lighter object relative to the fixed heavier object
    """
    # Calculate the offset of the heavier object from its fixed position
    offset = heavier_orbit - heavier_coord

    # Adjust the lighter object's orbit by subtracting this offset
    new_lighter_orbit = lighter_orbit - offset
    return new_lighter_orbit


def generate_orbit(
    problem: TwoBodyProblem,
    num_points_per_trajectory: int = 1_000,
    dt: float_type = 300,  # 5 minutes
    rng: int | Generator = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float_type]:
    """Generate the trajectories of the two objects in the two-body problem.
    Assumes the entire orbit is in the x-y plane, and that the eccentricity
    is less than 1 (i.e., elliptical orbit).

    Args:
        problem: TwoBodyProblem instance that contains the problem parameters.
        num_points_per_trajectory: Number of points to generate along each trajectory.
        dt: Time step in seconds between each point.
        rng: Random number generator or seed.

    Returns:
        Tuple of two numpy arrays representing the trajectories of the two objects,
        two numpy arrays representing the velocities of the two objects,
        and the eccentricity of the orbit.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # Total mass and graviational parameter
    mass_tot = problem.mass_1 + problem.mass_2
    mu = problem.gravitational_constant * mass_tot

    # Compute eccentricity vector
    pos_rel = generate_vector(problem.r_0_magnitude, problem.r_0_angle)
    vel_rel = generate_vector(problem.v_0_magnitude, problem.v_0_angle)
    sma, ecc = calculate_orbital_parameters(pos_rel, vel_rel, mu)

    # Generate time grid
    t_grid = np.arange(0, num_points_per_trajectory * dt, dt)

    # Calculate orbit
    mean_motion = np.sqrt(mu / sma**3)
   
    # --- Randomise starting point along the orbit ---
    orbital_period = 2 * np.pi / mean_motion  # years
    start_time_offset = rng.uniform(0, orbital_period)

    # Shift the mean anomaly by the time offset
    M_vals = mean_motion * (t_grid + start_time_offset)

    nu = np.array(
        [
            true_anomaly_from_anomaly(solve_kepler_equation(Mi, ecc), ecc)
            for Mi in M_vals
        ]
    )

    # Calculate orbit positions and velocities
    orbit_rel = np.zeros((num_points_per_trajectory, 3))
    orbit_vel_rel = np.zeros((num_points_per_trajectory, 3))
    for i, nui in enumerate(nu):
        orbit_rel[i] = compute_position(nui, sma, ecc)
        orbit_vel_rel[i] = compute_velocity(nui, sma, ecc, mu)

    # Rotate the orbit to match the initial position orientation
    rotation_matrix_r = np.array(
        [
            [np.cos(problem.r_0_angle), -np.sin(problem.r_0_angle), 0],
            [np.sin(problem.r_0_angle), np.cos(problem.r_0_angle), 0],
            [0, 0, 1],
        ]
    )
    orbit_rel = orbit_rel @ rotation_matrix_r.T

    # Rotate the velocities to match the initial velocity orientation
    rotation_matrix_v = np.array(
        [
            [np.cos(problem.v_0_angle), -np.sin(problem.v_0_angle), 0],
            [np.sin(problem.v_0_angle), np.cos(problem.v_0_angle), 0],
            [0, 0, 1],
        ]
    )
    orbit_vel_rel = orbit_vel_rel @ rotation_matrix_v.T

    # Project to x-y plane
    orbit_rel = orbit_rel[:, :2].reshape(-1, 2)
    orbit_vel_rel = orbit_vel_rel[:, :2].reshape(-1, 2)

    # Calculate the two objects' orbits
    orbit_1 = -orbit_rel * (problem.mass_2 / mass_tot) + np.array(
        problem.center_of_mass
    )
    orbit_2 = orbit_rel * (problem.mass_1 / mass_tot) + np.array(problem.center_of_mass)
    vel_1 = -orbit_vel_rel * (problem.mass_2 / mass_tot)
    vel_2 = orbit_vel_rel * (problem.mass_1 / mass_tot)

    return orbit_1, orbit_2, vel_1, vel_2, ecc


def generate_solar_system(
    eccs: list[float_type],
    smas: list[float_type],
    mass_1s: list[float_type],
    mass_2: float_type,
    seed: int = 0,
    gravitational_constant: float_type = 39.4784176044,  # AU^3/(year^2 * solar_mass)
    center_of_mass: Optional[np.ndarray] = None,
) -> MultipleTwoBodyProblem:
    """
    Generate a random MultipleTwoBodyProblem instance with a specified target eccentricities,
    semi-major axes, planet masses, and star mass.

    Args:
        eccs: List of eccentricities.
        smas: List of semi-major axes.
        mass_1s: List of planet masses.
        mass_2: Star mass.
        seed: Random number generator seed.
        gravitational_constant: Gravitational constant.
        center_of_mass: Center of mass of the system.
    Returns:
        A MultipleTwoBodyProblem instance with the specified eccentricities,
        semi-major axes, planet masses, and star mass.
    """
    for ecc in eccs:
        if ecc >= 1:
            raise ValueError("Only elliptical orbits are supported.")

    rng = np.random.default_rng(seed)
    # Sample 2D Gaussian center of mass
    if center_of_mass is None:
        center_of_mass = rng.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])

    r_0_magnitudes = []
    r_0_angles = []
    v_0_magnitudes = []
    v_0_angles = []
    for mass_1, sma, ecc in zip(mass_1s, smas, eccs):
        mass_tot = mass_1 + mass_2
        mu = gravitational_constant * mass_tot

        # Generate initial displacement at periapsis.
        r_0_magnitude = sma * (1 - ecc)
        r_0_angle = rng.uniform(0, 2 * np.pi)  # This is the tilt

        # v perpendicular to r
        v_0_angle = r_0_angle + np.pi / 2

        # Adjust v_0_magnitude to achieve the desired eccentricity
        v_squared = mu * (1 + ecc) / r_0_magnitude
        v_0_magnitude = np.sqrt(v_squared)

        r_0_magnitudes.append(r_0_magnitude)
        r_0_angles.append(r_0_angle)
        v_0_magnitudes.append(v_0_magnitude)
        v_0_angles.append(v_0_angle)

    return MultipleTwoBodyProblem(
        mass_1s=mass_1s,
        mass_2=mass_2,
        r_0_magnitudes=r_0_magnitudes,
        r_0_angles=r_0_angles,
        v_0_magnitudes=v_0_magnitudes,
        v_0_angles=v_0_angles,
        gravitational_constant=gravitational_constant,
        center_of_mass=center_of_mass,
    )


def generate_trajectories(
    problem: MultipleTwoBodyProblem,
    num_points_per_trajectory: int = 1_000,
    dt: float_type = 300,  # 5 minutes
    rng: int | Generator = 0,
    fix_heavier: bool = True,
) -> tuple[
    list[TrajectoryObservation],
    list[TrajectoryState],
    list[TrajectoryFunctionsOfState],
    TrajectoryObservation,
]:
    """Generate a trajectory with the heavier object fixed at some random coordinate, or not if fix_heavier=False.

    Args:
        problem: MultipleTwoBodyProblem instance that contains the problem parameters.
        num_points_per_trajectory: Number of points to generate along the orbit.
        dt: Time step in seconds between each point.
        rng: Random number generator or seed.
        fix_heavier: If True, fix the heavier object at a random coordinate. If False, do not fix (only for single planet).

    Returns:
        Tuple of lists of TrajectoryObservation, TrajectoryState, and TrajectoryFunctionsOfState instances.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    num_planets = len(problem.mass_1s)
    orbit_1s = []
    orbit_2s = []
    vel_1s = []
    vel_2s = []
    eccs = []
    for i in range(num_planets):
        assert (
            problem.mass_1s[i] < problem.mass_2
        ), f"Planet mass {problem.mass_1s[i]} must be smaller than star mass {problem.mass_2}"
        single_two_body_problem = TwoBodyProblem(
            mass_1=problem.mass_1s[i],
            mass_2=problem.mass_2,
            r_0_magnitude=problem.r_0_magnitudes[i],
            r_0_angle=problem.r_0_angles[i],
            v_0_magnitude=problem.v_0_magnitudes[i],
            v_0_angle=problem.v_0_angles[i],
            center_of_mass=problem.center_of_mass,
        )
        orbit_1, orbit_2, vel_1, vel_2, e = generate_orbit(
            single_two_body_problem, num_points_per_trajectory, dt, rng,
        )
        orbit_1s.append(orbit_1)
        orbit_2s.append(orbit_2)
        vel_1s.append(vel_1)
        vel_2s.append(vel_2)
        eccs.append(e)

    if fix_heavier:
        # Get shared location of sun.
        random_heavier_orbit = rng.choice(orbit_2s, size=1, axis=0).squeeze()
        heavier_coord = rng.choice(random_heavier_orbit, size=1, axis=0).squeeze()
        heavier_orbit = np.repeat(
            heavier_coord[np.newaxis, :], num_points_per_trajectory, axis=0
        )
    else:
        # Only valid for single planet
        assert num_planets == 1, "fix_heavier=False is only supported for single-planet problems."
        heavier_orbit = orbit_2s[0]

    trajectory_observations = []
    trajectory_states = []
    trajectory_functions_of_states = []

    for i in range(num_planets):
        orbit_1 = orbit_1s[i]
        orbit_2 = orbit_2s[i]
        vel_1 = vel_1s[i]
        vel_2 = vel_2s[i]
        e = eccs[i]
        lighter_orbit = orbit_1
        mass_heavy, mass_light = problem.mass_2, problem.mass_1s[i]
        vel_heavy, vel_light = vel_2, vel_1

        if fix_heavier:
            new_lighter_orbit = compute_orbit_in_new_frame(
                orbit_2, lighter_orbit, heavier_coord
            )
        else:
            # No frame shift; use original orbits
            new_lighter_orbit = lighter_orbit

        # Calculate relative velocity
        relative_velocity = vel_light - vel_heavy

        # Calculate the correct relative position vector
        relative_position = new_lighter_orbit - heavier_orbit

        # Calculate force vector using the correct relative position
        r = np.linalg.norm(relative_position, axis=1)
        force_magnitude = (
            problem.gravitational_constant * mass_heavy * mass_light / (r**2)
        )
        force_direction = -relative_position / r[:, np.newaxis]
        force_vector = force_magnitude[:, np.newaxis] * force_direction

        # Create return instances
        trajectory_observation = TrajectoryObservation(
            trajectory=new_lighter_orbit.tolist(),
        )
        trajectory_state = TrajectoryState(
            trajectory_light=new_lighter_orbit.tolist(),
            trajectory_heavy=heavier_orbit if fix_heavier else orbit_2,
            relative_velocity=relative_velocity.tolist(),
            m_light=mass_light,
            m_heavy=mass_heavy,
            gravitational_constant=problem.gravitational_constant,
            eccentricity=e,
        )
        trajectory_functions_of_state = TrajectoryFunctionsOfState(
            force_vector=force_vector.tolist(),
        )
        trajectory_observations.append(trajectory_observation)
        trajectory_states.append(trajectory_state)
        trajectory_functions_of_states.append(trajectory_functions_of_state)

    heavier_trajectory_observation = TrajectoryObservation(
        trajectory=heavier_orbit.tolist(),
    )

    return (
        trajectory_observations,
        trajectory_states,
        trajectory_functions_of_states,
        heavier_trajectory_observation,
    )
