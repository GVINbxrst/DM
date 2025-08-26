"""
Настройки приложения DiagMod
Конфигурация на основе переменных окружения
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения"""

    # Основные настройки приложения
    APP_NAME: str = "DiagMod"
    APP_VERSION: str = "1.0.0"
    APP_ENVIRONMENT: str = Field(default="development")  # env name matches field
    APP_DEBUG: bool = Field(default=False)

    # Настройки базы данных
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://diagmod_user:diagmod_password@localhost:5432/diagmod"
    )
    DATABASE_POOL_SIZE: int = Field(default=10)
    DATABASE_MAX_OVERFLOW: int = Field(default=20)

    # Настройки Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Настройки Celery
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")

    # Настройки API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_WORKERS: int = Field(default=4)
    API_RELOAD: bool = Field(default=False)

    # Настройки безопасности
    SECRET_KEY: str = Field(default="your-super-secret-key")
    JWT_SECRET_KEY: str = Field(default="your-jwt-secret-key")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30)

    # Настройки CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"]
    )

    # Настройки логирования
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")
    LOG_FILE_PATH: Optional[str] = Field(default="./logs/app.log")

    # Настройки обработки данных
    DATA_PATH: str = Field(default="./data")
    CSV_BATCH_SIZE: int = Field(default=10000)
    FEATURE_EXTRACTION_WORKERS: int = Field(default=4)
    # Включить параллельную обработку окон признаков (multiprocessing)
    FEATURE_EXTRACTION_PARALLEL: bool = Field(default=True)
    # Минимальное число окон, при котором имеет смысл включать параллельную обработку
    FEATURE_PARALLEL_MIN_WINDOWS: int = Field(default=4)
    MAX_CONCURRENT_FILES: int = Field(default=4)

    # Политика хранения сырых сигналов: если False – после успешного извлечения признаков
    # бинарные поля phase_a/phase_b/phase_c зануляются (экономия места)
    RETAIN_RAW_SIGNALS: bool = Field(default=True)

    # Ограничение выборки при обучении моделей аномалий (если None – без ограничения)
    ANOMALY_TRAIN_SAMPLE_SIZE: int | None = Field(default=None)
    # Включить стратифицированное семплирование по оборудованию
    ANOMALY_TRAIN_STRATIFIED: bool = Field(default=True)

    # Настройки ML моделей
    MODELS_PATH: str = Field(default="./models")
    MODEL_AUTO_RETRAIN: bool = Field(default=True)
    ANOMALY_THRESHOLD: float = Field(default=0.95)
    # Порог для статистического Z-score (MAD baseline) при онлайн-детекции аномалий
    ANOMALY_Z_THRESHOLD: float = Field(default=6.0)
    # Включить режим обработки только аномальных окон
    ANOMALIES_ONLY: bool = Field(default=False)
    # Какой онлайн-детектор использовать для фильтрации окон: stats | stream
    ONLINE_DETECTOR: str = Field(default="stats")
    # Количество окон для прогрева детектора, в течение которых не делаем отсев
    DETECTOR_WARMUP_WINDOWS: int = Field(default=20)
    # Кластеризация только по аномальным окнам (используется в админ-эндпоинтах)
    CLUSTERING_ANOMALIES_ONLY: bool = Field(default=False)
    # Кеширование моделей
    USE_MODEL_CACHE: bool = Field(default=True)
    # Кеш предсказаний (секунды)
    CACHE_EXPIRE_SECONDS: int = Field(default=300)

    # Параметры TCN / sequence forecasting
    TCN_ENABLED: bool = Field(default=True)
    TCN_WINDOW_SIZE: int = Field(default=32)
    TCN_PREDICTION_HORIZON: int = Field(default=4)
    TCN_CHANNELS: str = Field(default="64,64,64")  # список каналов через запятую
    TCN_KERNEL_SIZE: int = Field(default=3)
    TCN_DROPOUT: float = Field(default=0.1)
    TCN_MAX_EPOCHS: int = Field(default=10)
    TCN_LEARNING_RATE: float = Field(default=1e-3)

    # Настройки сигналов
    SIGNAL_SAMPLING_RATE: int = Field(default=25600)
    FFT_WINDOW_SIZE: int = Field(default=1024)
    OVERLAP_RATIO: float = Field(default=0.5)

    # Настройки мониторинга
    PROMETHEUS_ENABLED: bool = Field(default=True)
    PROMETHEUS_PORT: int = Field(default=8001)

    # Почасовые агрегаты / feature store
    FEATURE_SUMMARY_ENABLED: bool = Field(default=True)
    FEATURE_STORE_BACKEND: str = Field(default="db")  # db | redis
    FEATURE_SUMMARY_WINDOW: int = Field(default=60)  # минут, резерв (пока используем только час)
    MAX_FEATURE_STORE_POINTS: int = Field(default=2000)  # лимит точек в памяти (redis backend)

    # Авто-инжест при старте API (опционально)
    INGEST_STARTUP_DIR: Optional[str] = Field(default=None)
    INGEST_FILE_PATTERN: str = Field(default="*.csv")
    INGEST_MAX_FILES: Optional[int] = Field(default=None)
    INGEST_TRAIN_FULL: bool = Field(default=False)
    AUTO_FORECAST_AFTER_INGEST: bool = Field(default=True)

    # Настройки Dashboard
    DASHBOARD_HOST: str = Field(default="0.0.0.0")
    DASHBOARD_PORT: int = Field(default=8501)
    API_URL: str = Field(default="http://localhost:8000")

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        vu = v.upper()
        if vu not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return vu

    @field_validator("APP_ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = ["development", "staging", "production", "test"]
        vl = v.lower()
        if vl not in valid_envs:
            raise ValueError(f"APP_ENVIRONMENT must be one of {valid_envs}")
        return vl

    @property
    def is_development(self) -> bool:
        return self.APP_ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        return self.APP_ENVIRONMENT == "production"

    @property
    def is_testing(self) -> bool:
        return self.APP_ENVIRONMENT == "test"

    @property
    def data_path(self) -> Path:
        return Path(self.DATA_PATH)

    @property
    def models_path(self) -> Path:
        return Path(self.MODELS_PATH)

    @models_path.setter  # type: ignore
    def models_path(self, value: Path | str):  # Позволяет monkeypatch в тестах
        # Pydantic BaseSettings неизменяема по умолчанию, но тест использует monkeypatch.setattr.
        # Сохраняем совместимость, устанавливая базовое поле.
        object.__setattr__(self, 'MODELS_PATH', str(value))

    @property
    def log_file_path(self) -> Optional[Path]:
        return Path(self.LOG_FILE_PATH) if self.LOG_FILE_PATH else None

    # Совместимость с кодом, использующим «snake_case» атрибуты
    @property
    def log_level(self) -> str:  # pragma: no cover - простое свойство
        return self.LOG_LEVEL

    @property
    def log_format(self) -> str:  # pragma: no cover
        return self.LOG_FORMAT

    @property
    def environment(self) -> str:  # pragma: no cover
        return self.APP_ENVIRONMENT

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Игнорируем дополнительные переменные окружения
        protected_namespaces=()  # Разрешаем имена полей с префиксом model_
    )


@lru_cache()
def get_settings() -> Settings:
    """Получить настройки приложения (с кешированием)"""
    return Settings()
