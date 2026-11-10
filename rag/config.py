"""프로젝트 전체 설정. 값은 .env 또는 환경변수에서 읽고, 없으면 기본값을 쓴다."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # API 키 (비어 있으면 실제 호출 시점에 에러가 난다)
    anthropic_api_key: str = ""
    voyage_api_key: str = ""

    # 모델
    llm_model: str = "claude-haiku-5-5"
    embedding_model: str = "voyage-3.5"  # 1024차원

    # 경로
    pdf_path: Path = ROOT / "data" / "catalog.pdf"
    chroma_dir: Path = ROOT / "chroma_db"
    prompt_path: Path = ROOT / "prompts" / "answer_v1.txt"

    # 청킹 · 검색 (3주차에 실험할 값들)
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4


settings = Settings()
