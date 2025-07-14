from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Checkpoint directories
CKPT_DIR = BASE_DIR / "checkpoints"
PHYSICS_CKPT_DIR = CKPT_DIR / "physics"
GRIDWORLD_CKPT_DIR = CKPT_DIR / "gridworld"
OTHELLO_CKPT_DIR = CKPT_DIR / "othello"

# Config directories
CONFIG_DIR = BASE_DIR / "configs"
PHYSICS_CONFIG_DIR = CONFIG_DIR / "physics"
GRIDWORLD_CONFIG_DIR = CONFIG_DIR / "gridworld"
OTHELLO_CONFIG_DIR = CONFIG_DIR / "othello"

# Data directories
# If you have an ephemeral drive, you can set this to the path of the ephemeral drive.
DATA_DIR = Path("/nvme/data").resolve()
# If you don't have an ephemeral drive, you can set this to the path of the BASE_DIR.
# DATA_DIR = BASE_DIR / "data"
PHYSICS_DATA_DIR = DATA_DIR / "physics"
GRIDWORLD_DATA_DIR = DATA_DIR / "gridworld"
OTHELLO_DATA_DIR = DATA_DIR / "othello"

# Extrapolation directories
EXT_DIR = BASE_DIR / "extrapolations"
PHYSICS_EXT_DIR = EXT_DIR / "physics"
GRIDWORLD_EXT_DIR = EXT_DIR / "gridworld"
OTHELLO_EXT_DIR = EXT_DIR / "othello"
