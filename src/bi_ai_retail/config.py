from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_URL = "https://raw.githubusercontent.com/dbdmg/data-science-lab/master/datasets/online_retail.csv"
RAW_DATA_PATH = RAW_DIR / "online_retail.csv"

TRAIN_CUTOFF_DATE = "2011-10-01"
FORECAST_HORIZON_DAYS = 30

