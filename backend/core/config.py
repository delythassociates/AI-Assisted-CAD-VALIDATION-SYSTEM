from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./eureka.db"
    GNN_MODEL_PATH: str = "F:/Varroc/data/models/gnn_model_real_cpu.pt"
    GNN_THRESHOLD_PATH: str = "F:/Varroc/data/models/threshold_real.txt"
    SW_VERSION: str = "2025"
    OPA_ENABLED: bool = False
    OPA_ENDPOINT: str = "http://localhost:8181"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    LOG_LEVEL: str = "INFO"
    FEAT_DATA_DIR: str = "data/abc_raw/feat"
    STAT_DATA_DIR: str = "data/abc_raw/stat"

    model_config = {"env_prefix": "EUREKA_"}

settings = Settings()
