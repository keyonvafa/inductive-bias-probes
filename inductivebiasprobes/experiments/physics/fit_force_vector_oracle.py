import logging
from pathlib import Path

import numpy as np
import yaml
import os

from inductivebiasprobes.paths import (
    PHYSICS_CONFIG_DIR,
    PHYSICS_DATA_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open(PHYSICS_CONFIG_DIR / "force_vector_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

train_state_file = "state_solar_system_two_body.npy"
train_force_file = "force_vector_solar_system_two_body.npy"
test_state_file = "state_solar_system_two_body.npy"
test_force_file = "force_vector_solar_system_two_body.npy"

train_state_data = np.load(PHYSICS_DATA_DIR / train_state_file)
train_force_data = np.load(PHYSICS_DATA_DIR / train_force_file)
test_state_data = np.load(PHYSICS_DATA_DIR / test_state_file)
test_force_data = np.load(PHYSICS_DATA_DIR / test_force_file)

odd_indices_train = np.arange(1, train_state_data.shape[1], 2)
odd_indices_test = np.arange(1, test_state_data.shape[1], 2)

train_state_data = train_state_data[:, odd_indices_train]
train_force_data = train_force_data[:, odd_indices_train]
test_state_data = test_state_data[:, odd_indices_test]
test_force_data = test_force_data[:, odd_indices_test]

# Unmask 10 timesteps per sequence, 
unmasked_per_sequence = 10
rs = np.random.RandomState(0)
sample_mask = np.zeros((train_state_data.shape[0], train_state_data.shape[1]))
for i in range(train_state_data.shape[0]):
    sample_inds = rs.choice(train_state_data.shape[1], size=unmasked_per_sequence, replace=False)
    sample_mask[i, sample_inds] = 1

# Only include the unmasked data
train_state_data = train_state_data[sample_mask == 1]
train_force_data = train_force_data[sample_mask == 1]
test_state_data = test_state_data.reshape(-1, test_state_data.shape[-1])
test_force_data = test_force_data.reshape(-1, test_force_data.shape[-1])

# Trying KNN
X_train_full = train_state_data
Y_train_full = train_force_data
X_test_full = test_state_data
Y_test_full = test_force_data

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_full, Y_train_full)

# Get predictions for the solar system
solar_system_state_data = np.load(PHYSICS_DATA_DIR / "state_solar_system_two_body.npy")
odd_indices_solar_system = np.arange(1, solar_system_state_data.shape[1], 2)
solar_system_state_data = solar_system_state_data[:, odd_indices_solar_system]
true_solar_system_force_vectors = np.load(PHYSICS_DATA_DIR / "force_vector_solar_system_two_body.npy")

solar_system_preds = knn_model.predict(solar_system_state_data.reshape(-1, solar_system_state_data.shape[-1]))
solar_system_preds = solar_system_preds.reshape(solar_system_state_data.shape[0], -1, 2)

for planet_idx in range(solar_system_state_data.shape[0]):
    os.makedirs(f"scratch/oracle", exist_ok=True)
    np.save(f"scratch/oracle/force_vector_oracle_preds_{planet_idx}.npy", solar_system_preds[planet_idx, :-1])
    np.save(f"scratch/oracle/force_vector_oracle_truth_{planet_idx}.npy", true_solar_system_force_vectors[planet_idx][odd_indices_solar_system][:-1])



