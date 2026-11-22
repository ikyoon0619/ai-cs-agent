# rag/store.py
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from rag.load import PDF_PATH, load_pdf
from rag.split import split_docs
from rag.embed import get_embeddings

load_dotenv()                          # .env 파일 내용을 환경변수로 읽어옴
DB_URL = os.getenv("DATABASE_URL")     # postgresql+psycopg://... (지난주 .env에 넣은 값)
COLLECTION = "catalog"                 # 저장 묶음 이름 (Chroma 때와 같은 이름 유지)


def build_store():
    """PDF → 청크 → 벡터 → PostgreSQL 저장. 처음 한 번, 또는 PDF가 바뀌었을 때만 실행."""
    pages = load_pdf(PDF_PATH)         # 1일차: PDF를 페이지 단위로 읽기
    chunks = split_docs(pages)         # 2일차: 500자 단위로 자르기
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i   # PDF 안 순서 번호. DB에서 꺼낼 때 같은 순서로 정렬하려고
    return PGVector.from_documents(
        documents=chunks,              # 저장할 청크들
        embedding=get_embeddings(),    # 3일차: Voyage로 각 청크를 1024차원 벡터로 변환
        collection_name=COLLECTION,
        connection=DB_URL,             # 어느 DB에 저장할지
        pre_delete_collection=True,    # 같은 이름 묶음이 있으면 지우고 새로 저장 (중복 방지)
        use_jsonb=True,                # page_label 같은 메타데이터를 JSONB 컬럼에 저장
    )


def load_store():
    """이미 저장된 것을 열기만 함. 임베딩 비용 0원."""
    return PGVector(
        embeddings=get_embeddings(),   # 질문을 벡터로 바꿀 때 씀. Chroma는 embedding_function
        collection_name=COLLECTION,
        connection=DB_URL,
        use_jsonb=True,
    )


def load_chunks():
    """DB에 저장된 청크를 전부 꺼낸다 (BM25용). PDF 파일 없이도 동작 → 배포 서버에 PDF를 안 올려도 됨."""
    sql = text("""
               SELECT e.document, e.cmetadata
               FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
               WHERE c.name = :name
               ORDER BY (e.cmetadata->>'chunk_index')::int NULLS LAST, e.id
               """)
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"name": COLLECTION}).all()
    engine.dispose()                   # 한 번 쓰고 끝이니 연결 정리
    return [Document(page_content=doc, metadata=meta) for doc, meta in rows]


def get_store():
    """앱(answer.py)이 부르는 함수. 저장은 ingest 스크립트가 따로 하므로 여기선 열기만."""
    return load_store()


if __name__ == "__main__":
    store = get_store()

    questions = [
        "연비가 어떻게 돼?",
        "트림 종류 알려줘",
        "안전 사양 뭐 있어?",
    ]

    for question in questions:
        print("==== 질문:", question)
        results = store.similarity_search(question, k=3)

        for doc in results:
            print("페이지:", doc.metadata["page_label"])
            print("본문:", doc.page_content[:80])
            print("---")