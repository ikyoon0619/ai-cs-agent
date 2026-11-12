# rag/store.py
import os

from langchain_chroma import Chroma

from rag.load import PDF_PATH, load_pdf
from rag.split import split_docs
from rag.embed import get_embeddings

DB_DIR = "chroma_db"         # 저장할 폴더 이름
COLLECTION = "catalog"       # DB 안의 테이블 이름 같은 것


def build_store():
    # 처음 한 번: PDF → 청크 → 좌표로 바꿔서 저장
    pages = load_pdf(PDF_PATH)
    chunks = split_docs(pages)                 # 빈칸 ①: 2일차에 만든 함수
    return Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),                # 빈칸 ②: 오늘 만든 변환기 함수
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
    )

def load_store():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
    )

def get_store():
    # 폴더가 있으면 열고, 없으면 새로 만든다
    if os.path.exists(DB_DIR):
        return load_store()
    return build_store()

if __name__ == "__main__":
    store = get_store()

    questions = [
        "연비가 어떻게 돼?",
        "트림 종류 알려줘",
        "안전 사양 뭐 있어?",
    ]

    for question in questions:                 # 질문을 하나씩 꺼내서
        print("==== 질문:", question)
        results = store.similarity_search(question, k=3)

        for doc in results:                    # 찾은 청크 3개를 하나씩 출력
            print("페이지:", doc.metadata["page_label"])
            print("본문:", doc.page_content[:80])
            print("---")