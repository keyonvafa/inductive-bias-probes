import logging
from pathlib import Path

import numpy as np
import pysr
import torch
import tqdm
import yaml

from inductivebiasprobes.paths import (
    PHYSICS_CKPT_DIR,
    PHYSICS_CONFIG_DIR,
    PHYSICS_DATA_DIR,
)
from inductivebiasprobes import ModelConfig, Model
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(pretrained):
    ckpt_path = PHYSICS_CKPT_DIR / "gpt" / pretrained / "ckpt.pt"
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


def top_k_closest_pairs_2d(states_test: np.ndarray,
                        states_train: np.ndarray,
                        k: int = 100):
    """Use this to make sure that states used for inference are close to the training states.
    """
    train_flat = states_train
    test_flat  = states_test
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
    nn.fit(train_flat)
    distances, neighbor_idx = nn.kneighbors(test_flat, return_distance=True)
    distances, neighbor_idx = distances.ravel(), neighbor_idx.ravel()
    top_idx_test = np.argsort(distances)[:k]
    top_idx_train = neighbor_idx[top_idx_test]
    return top_idx_test, top_idx_train, distances[top_idx_test]



def main():
    # Do everything on train because we're doing the masking thing. 
    train_state_file = "full_state_two_body_train.npy"
    train_force_file = "force_magnitude_two_body_train.npy"
    train_force_file_masked = "force_magnitude_two_body_train_masked.npy"
    test_obs_file = "obs_two_body_train.npy"

    # First, fit the oracle model (KNN). 
    logger.info("[Fitting symbolic regression on an oracle model]")

    with open(PHYSICS_CONFIG_DIR / "force_magnitude_config.yaml") as f:
        force_magnitude_config = yaml.load(f, Loader=yaml.FullLoader)

    scratch_dir = Path("scratch")
    scratch_dir.mkdir(exist_ok=True)

    # Load data
    all_state_data = np.load(PHYSICS_DATA_DIR / train_state_file)[:, :-1, :]
    all_force_data = np.load(PHYSICS_DATA_DIR / train_force_file)[:, :-1, :]
    all_force_data_masked = np.load(PHYSICS_DATA_DIR / train_force_file_masked)[:, :-1, :]
    # Only include odd indices
    all_state_data = all_state_data[:, 1::2]
    all_force_data = all_force_data[:, 1::2]
    all_force_data_masked = all_force_data_masked[:, 1::2]
    # Get state dimension
    state_dim = all_state_data.shape[-1]
    mask_id = force_magnitude_config["mask_id"]
    num_train = 9_000
    states_test = all_state_data[num_train:]
    forces_test = all_force_data[num_train:]
    # Flatten train
    states_train = all_state_data[:num_train].reshape(-1, state_dim)
    forces_train = all_force_data[:num_train].reshape(-1)
    # Instead of randomly sampling indices for inference, use indices such that 
    # the states used for inference are close to the training states. 
    closest_test_inds, closest_train_inds, dists = top_k_closest_pairs_2d(states_test.reshape(-1, state_dim), states_train, k=5000)
    unmasked_test_idx = np.where(all_force_data_masked[num_train:] != float("inf"))[1].reshape(-1, 2); assert len(unmasked_test_idx) == len(states_test)
    # Add unmasked test
    unmasked_states_from_test = np.concatenate([states_test[np.arange(len(states_test)), unmasked_test_idx[:, 0]], states_test[np.arange(len(states_test)), unmasked_test_idx[:, 1]]])
    unmasked_forces_from_test = np.concatenate([forces_test[np.arange(len(forces_test)), unmasked_test_idx[:, 0]], forces_test[np.arange(len(forces_test)), unmasked_test_idx[:, 1]]])[:, 0]
    states_train = np.concatenate([states_train, unmasked_states_from_test], axis=0)
    forces_train = np.concatenate([forces_train, unmasked_forces_from_test], axis=0)
    # Flatten test
    states_test = states_test.reshape(-1, state_dim)
    forces_test = forces_test.reshape(-1)
    test_idx = closest_test_inds
    states_test = states_test[test_idx]
    forces_test = forces_test[test_idx]

    # ----------------------
    # Load test split (for symbolic regression)
    # ----------------------
    input_data_test = np.load(PHYSICS_DATA_DIR / test_obs_file)[:, :-1, :]
    input_data_test = input_data_test[num_train:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rs = np.linalg.norm(states_test[:, :2], axis=1)
    m1s = states_test[:, 4]
    m2s = states_test[:, 5]

    logger.info("[Fitting symbolic regression on a GPT model]")
    pretrained_dir = "next_token_pt_force_magnitude_transfer"
    gpt_model = load_model(pretrained_dir)
    gpt_model.to(device)
    gpt_model.eval()
    # Process GPT predictions in batches
    batch_size = 100  
    gpt_predictions = []
    with torch.no_grad():
        input_data_test_tensor = torch.from_numpy(input_data_test).long().to(device)
        num_batches = int(np.ceil(len(input_data_test_tensor) / batch_size))
        for i in tqdm.trange(num_batches, desc="Processing GPT predictions"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(input_data_test_tensor))
            batch_input = input_data_test_tensor[start_idx:end_idx]
            batch_logits = gpt_model(batch_input)
            batch_predictions = batch_logits
            batch_predictions_np = batch_predictions.detach().cpu().numpy()
            gpt_predictions.append(batch_predictions_np)
    
    gpt_predictions = np.concatenate(gpt_predictions)
    # Only include non-sun indices. 
    gpt_predictions_odd = gpt_predictions[:, 1::2]
    gpt_predictions_flat = gpt_predictions_odd.reshape(-1)
    gpt_test_predictions = gpt_predictions_flat[test_idx]

    gpt_equations = []
    for seed in range(5):
        idx = np.random.RandomState(seed).choice(states_test.shape[0], 1_000, replace=False)
        X_sr = np.stack([rs[idx], m1s[idx], m2s[idx]], axis=1)
        y_sr_gpt = gpt_test_predictions[idx]
        best_equation = None
        best_score = 0.0
        for restart_seed in [0, 1, 2]:
            gpt_pysr_model = pysr.PySRRegressor(
                niterations=100,
                binary_operators=["+", "*"],
                unary_operators=["sin", "cos", "exp", "inv(x) = 1/x"],
                extra_sympy_mappings={"inv": lambda x: 1 / x},
                elementwise_loss="loss(prediction, target) = max(1e-8, abs(prediction - target)) - 1e-8",
                maxsize=20,
                random_state=restart_seed,
                parallelism="serial",
                deterministic=True,
                model_selection="score",
            )
            gpt_pysr_model.fit(X_sr, y_sr_gpt)
            best_model = gpt_pysr_model.get_best()
            if best_model is not None and best_model.score > best_score:
                best_score = best_model.score
                best_equation = best_model.equation
        logger.info(f"[Best GPT equation: {best_equation}]")
        gpt_equations.append(best_equation)

    np.save(scratch_dir / "gpt_equations.npy", gpt_equations)
    print(f"The following equations were found for the GPT model: {gpt_equations}")

    # ----------------------
    # Try oracle KNN model.
    # ----------------------

    print("Fitting oracle model...")

    # Scale the features for better KNN performance
    scaler = StandardScaler()
    states_train_scaled = scaler.fit_transform(states_train)
    states_test_scaled = scaler.transform(states_test)
    
    knn = KNeighborsRegressor(n_neighbors=2, n_jobs=-1)
    knn.fit(states_train_scaled, forces_train)
    force_predictions = knn.predict(states_test_scaled) 

    oracle_equations = []
    for seed in range(5):
        idx = np.random.RandomState(seed).choice(states_test.shape[0], 1_000, replace=False)
        X_sr = np.stack([rs[idx], m1s[idx], m2s[idx]], axis=1)
        y_sr = force_predictions[idx]
        best_equation = None
        best_score = 0.0
        for restart_seed in [0, 1, 2]:
            pysr_model = pysr.PySRRegressor(
                niterations=100,   
                binary_operators=["+", "*"],
                unary_operators=["sin", "cos", "exp", "inv(x) = 1/x"],
                extra_sympy_mappings={"inv": lambda x: 1 / x},
                elementwise_loss="loss(prediction, target) = max(1e-8, abs(prediction - target)) - 1e-8",
                maxsize=20,
                random_state=restart_seed,      
                parallelism="serial", 
                deterministic=True,   
                model_selection="score",
            )
            pysr_model.fit(X_sr, y_sr)
            best_model = pysr_model.get_best()
            if best_model is not None and best_model.score > best_score:
                best_score = best_model.score
                best_equation = best_model.equation
        logger.info(f"[Best equation: {best_equation}]")
        oracle_equations.append(best_equation)
    np.save(scratch_dir / "oracle_equations.npy", oracle_equations)
    print(f"The following equations were found for the oracle model: {oracle_equations}")


if __name__ == "__main__":
    main()
