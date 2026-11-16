# scripts/ingest.py
# 카탈로그를 PostgreSQL에 저장하는 스크립트. PDF가 바뀔 때만 다시 실행.
from rag.store import build_store

if __name__ == "__main__":
    build_store()
    print("저장 완료")