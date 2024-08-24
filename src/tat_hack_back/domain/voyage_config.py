from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VoyageApiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='VOYAGE_API_')

    KEY: str = Field(...)


voyage_api_config = VoyageApiConfig()
