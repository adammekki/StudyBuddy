from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Settings:
    rag_dir: Path = Path("RAG")
    data_dir: Path = Path("data")
    gen_model: str = "gemma4:e4b"
    judge_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"
    k: int = 8
    threshold: float = 0.35
    chunk_size: int = 800
    chunk_overlap: int = 100


def load_settings() -> Settings:
    rag_dir = Path(os.environ.get("STUDYBUDDY_RAG_DIR", "RAG"))
    data_dir = Path(os.environ.get("STUDYBUDDY_DATA_DIR", "data"))
    settings = Settings(rag_dir=rag_dir, data_dir=data_dir)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.rag_dir.mkdir(parents=True, exist_ok=True)
    return settings
