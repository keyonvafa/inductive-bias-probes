#!/usr/bin/env python3
"""
Test script to compare the output of original and fixed generate_data.py
Generates 1000 data points with both versions and verifies they produce identical results.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add the project to path
sys.path.insert(0, '/scratch/u5ea/bogao357.u5ea/inductive-bias-probes')

import pandas as pd
from inductivebiasprobes import (
    build_exoplanet_distributions,
    build_two_body_distributions,
)

# Import both versions of the generation functions
# We need to import them with different names
import importlib.util

def load_module(name, path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load original and fixed versions
original_module = load_module(
    "generate_data_original",
    "/scratch/u5ea/bogao357.u5ea/inductive-bias-probes/inductivebiasprobes/experiments/physics/generate_data.py"
)
fixed_module = load_module(
    "generate_data_fixed", 
    "/scratch/u5ea/bogao357.u5ea/inductive-bias-probes/inductivebiasprobes/experiments/physics/generate_data_fixed.py"
)


def run_test(num_trajectories=1000, num_points_per_trajectory=100, num_bins=7000, seed=42):
    """Generate data with both versions and compare."""
    
    print(f"Testing with {num_trajectories} trajectories, {num_points_per_trajectory} points each")
    print(f"Seed: {seed}, num_bins: {num_bins}")
    print("-" * 60)
    
    # Create temporary directories for output
    original_dir = Path(tempfile.mkdtemp(prefix="original_"))
    fixed_dir = Path(tempfile.mkdtemp(prefix="fixed_"))
    
    try:
        # Load exoplanet distributions
        exoplanet_df = pd.read_csv(
            "/scratch/u5ea/bogao357.u5ea/inductive-bias-probes/inductivebiasprobes/experiments/physics/exoplanet_data.csv"
        )
        exoplanet_distributions = build_exoplanet_distributions(exoplanet_df, max_sma=32.0)
        
        dts = [7 * 0.0027, 6 * 30 * 0.0027]
        dt_to_token = {0: num_bins, 1: num_bins + 1}
        
        print("Generating data with ORIGINAL version...")
        original_seed = original_module.generate_data_parallel(
            exoplanet_distributions=exoplanet_distributions,
            num_points_per_trajectory=num_points_per_trajectory,
            num_trajectories=num_trajectories,
            dts=dts,
            dt_to_token=dt_to_token,
            seed=seed,
            num_workers=4,  # Use fewer workers for test
            out_dir=original_dir,
            label="test",
            num_bins=num_bins,
            skip_states=True,  # Skip states for faster test
            debug=True,  # Use debug mode (no multiprocessing) for deterministic results
        )
        print(f"  Original returned seed: {original_seed}")
        
        print("Generating data with FIXED version...")
        fixed_seed = fixed_module.generate_data_parallel(
            exoplanet_distributions=exoplanet_distributions,
            num_points_per_trajectory=num_points_per_trajectory,
            num_trajectories=num_trajectories,
            dts=dts,
            dt_to_token=dt_to_token,
            seed=seed,
            num_workers=4,
            out_dir=fixed_dir,
            label="test",
            num_bins=num_bins,
            skip_states=True,
            debug=True,
        )
        print(f"  Fixed returned seed: {fixed_seed}")
        
        # Compare outputs
        print("\n" + "=" * 60)
        print("COMPARING OUTPUTS")
        print("=" * 60)
        
        # Load and compare observation files
        original_obs = np.load(original_dir / "obs_test.npy")
        fixed_obs = np.load(fixed_dir / "obs_test.npy")
        
        print(f"\nObservation shape - Original: {original_obs.shape}, Fixed: {fixed_obs.shape}")
        
        if original_obs.shape == fixed_obs.shape:
            print("  ✓ Shapes match")
        else:
            print("  ✗ Shapes DO NOT match!")
            return False
        
        if np.array_equal(original_obs, fixed_obs):
            print("  ✓ All values are IDENTICAL")
        else:
            diff_count = np.sum(original_obs != fixed_obs)
            total_count = original_obs.size
            print(f"  ✗ Values differ! {diff_count}/{total_count} elements are different ({100*diff_count/total_count:.2f}%)")
            
            # Show some differences
            if diff_count > 0:
                diff_indices = np.where(original_obs != fixed_obs)
                print(f"  First 5 differences:")
                for i in range(min(5, len(diff_indices[0]))):
                    idx = tuple(d[i] for d in diff_indices)
                    print(f"    Index {idx}: original={original_obs[idx]}, fixed={fixed_obs[idx]}")
            return False
        
        # Compare num_bodies files
        original_nb = np.load(original_dir / "num_bodies_test.npy")
        fixed_nb = np.load(fixed_dir / "num_bodies_test.npy")
        
        print(f"\nNum bodies shape - Original: {original_nb.shape}, Fixed: {fixed_nb.shape}")
        
        if np.array_equal(original_nb, fixed_nb):
            print("  ✓ All values are IDENTICAL")
        else:
            print("  ✗ Values differ!")
            return False
        
        # Compare returned seeds
        print(f"\nReturned seeds - Original: {original_seed}, Fixed: {fixed_seed}")
        if original_seed == fixed_seed:
            print("  ✓ Seeds match")
        else:
            print("  ✗ Seeds DO NOT match!")
            return False
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED - OUTPUTS ARE IDENTICAL")
        print("=" * 60)
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(original_dir, ignore_errors=True)
        shutil.rmtree(fixed_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test consistency between original and fixed generate_data.py")
    parser.add_argument("--num_trajectories", type=int, default=1000, help="Number of trajectories to generate")
    parser.add_argument("--num_points", type=int, default=100, help="Number of points per trajectory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    success = run_test(
        num_trajectories=args.num_trajectories,
        num_points_per_trajectory=args.num_points,
        seed=args.seed,
    )
    sys.exit(0 if success else 1)
