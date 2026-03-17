from pathlib import Path

# Project paths
PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = PROJECT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "models"
TABLE_DIR = PROJECT_ROOT / "tables"

# Zenodo
ZENODO_RECORD_ID = "4536377"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
TARGET_FILENAME = "events_anomalydetection.h5"
HDF_KEY = "df"

# CWoLa default regions
SR_LOW = 3300
SR_HIGH = 3700

SB_LEFT_LOW = 2300
SB_LEFT_HIGH = 3000

SB_RIGHT_LOW = 4000
SB_RIGHT_HIGH = 4700

# Model / training
RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE_FROM_TEMP = 0.50

BATCH_SIZE = 2048
EPOCHS = 40
LEARNING_RATE = 1e-3

FEATURE_COLS = [
    "ptj1", "ptj2",
    "mj1", "mj2",
    "tau21_j1", "tau32_j1",
    "tau21_j2", "tau32_j2",
    "pt_balance", "m_ratio",
    "pxj1", "pyj1", "pzj1",
    "pxj2", "pyj2", "pzj2"
]
