# rag/split.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag.load import PDF_PATH, load_pdf     # 어제 만든 함수를 재사용

def split_docs(docs: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)          # Document 리스트를 받아 Document 리스트로 나눠 주는 메서드

if __name__ == "__main__":
    pages = load_pdf(PDF_PATH)
    chunks = split_docs(pages)
    print("페이지 수:", len(pages))
    print("청크 수:", len(chunks))
    print("3번째 청크 metadata:", chunks[2].metadata)
    print("3번째 청크 본문:", chunks[2].page_content)