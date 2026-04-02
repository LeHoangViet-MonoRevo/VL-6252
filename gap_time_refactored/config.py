# config.py

TARGET_USER_MAP = {
    "teknia": 86,
    "daiho-gr": 90,
    "kondosk": 88,
    "topura": 74,
    "stertec": 116,
    "yagi-kinzoku": 72,
    "kimurass": 128,
}
TARGET_ORGS = list(TARGET_USER_MAP.values())

PRODUCTION_LOTS_PATH = "./data_prod/production/production_lots.csv"
PLANNING_PATH = "./data_prod/production/planning_processes.csv"

GAP_THRESHOLD_HOURS = 30 * 24  # 30 days

PRODUCTION_LOTS_DT_COLS = [
    "delivery_deadline",
    "deadline",
    "material_deadline",
    "start_at",
    "end_at",
    "original_start_date",
    "start_date",
    "created_at",
    "updated_at",
    "deleted_at",
]
PLANNING_DT_COLS = [
    "deadline",
    "start_at",
    "end_at",
    "created_at",
    "updated_at",
    "deleted_at",
]
