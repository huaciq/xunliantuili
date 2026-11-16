from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Vision Training Platform"
    app_env: str = "development"
    app_secret_key: str = "development-secret-change-me"
    access_token_expire_minutes: int = 480
    database_url: str = "sqlite:///./platform.db"
    redis_url: str = "redis://localhost:6379/0"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "train-platform"
    minio_secret_key: str = "change-this-minio-password"
    minio_secure: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    storage_root: str = "./storage"
    max_upload_bytes: int = 5 * 1024 * 1024 * 1024
    max_archive_entries: int = 200_000
    max_extracted_bytes: int = 100 * 1024 * 1024 * 1024
    executor_backend: Literal["fake", "docker"] = "fake"
    fake_run_duration_seconds: int = 12
    scheduler_poll_seconds: float = 0.5
    docker_binary: str = "docker"
    docker_network: str = "bridge"
    docker_user: str = "1000:1000"
    docker_cpu_limit: float = 4.0
    docker_memory_limit_gb: int = 16
    docker_shm_size_gb: int = 4
    docker_pids_limit: int = 1024
    nvidia_smi_binary: str = "nvidia-smi"
    max_collected_artifacts: int = 200
    max_collected_artifact_bytes: int = 20 * 1024 * 1024 * 1024
    bootstrap_admin_email: str = "admin@example.com"
    bootstrap_admin_password: str = "ChangeMe123!"
    bootstrap_admin_name: str = "Platform Admin"
    auto_create_tables: bool = True
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
