from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SGIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='ROSPATENT_SCRAPER_APP_')

    HTTP_PROTOCOL: str = Field(env="HTTP_PROTOCOL", default="http")
    HOST: str = Field(env="HOST", default="0.0.0.0")
    PORT: int = Field(env="PORT", default=8081)

    WORKERS_COUNT: int = Field(env="WORKERS_COUNT", default=1)

    AUTO_RELOAD: bool = Field(env="AUTORELOAD", default=True)
    TIMEOUT: int = Field(env="TIMEOUT", default=420)

    WSGI_APP: str = Field(env="WSGI_APP", default="rospatent_scraper.api.main:app")
    WORKER_CLASS: str = Field(env="WORKER_CLASS", default="uvicorn.workers.UvicornWorker")


sgi_config = SGIConfig()
