"""step 1. PDF를 페이지 단위 Document 리스트로 읽는다."""

from pathlib import Path

from langchain_core.documents import Document


def load_pdf(path: Path) -> list[Document]:
    # TODO: PyPDFLoader로 path를 읽어서 돌려주세요.
    #   힌트: from langchain_community.document_loaders import PyPDFLoader
    #   metadata["page"]는 0부터 시작합니다.
    raise NotImplementedError


if __name__ == "__main__":
    from rag.config import settings

    docs = load_pdf(settings.pdf_path)
    print(f"페이지 수: {len(docs)}")
    print(docs[0].metadata)
