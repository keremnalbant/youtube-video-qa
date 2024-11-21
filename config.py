import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "")
    LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "")
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "")
    WEAVIATE_AUTH_KEY: str = os.getenv("WEAVIATE_AUTH_KEY", "")
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY", "")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY", "")

    def __post_init__(self):
        required_vars = [
            "WEAVIATE_URL",
            "WEAVIATE_AUTH_KEY",
            "AWS_ACCESS_KEY",
            "AWS_SECRET_KEY",
        ]
        missing_vars = [var for var in required_vars if getattr(self, var) is None]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )


config = Config()
