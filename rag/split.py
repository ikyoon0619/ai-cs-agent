"""step 2. 페이지를 검색하기 좋은 크기의 청크로 나눈다."""

from langchain_core.documents import Document


def split_docs(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    # TODO: RecursiveCharacterTextSplitter로 나누세요.
    #   힌트: from langchain_text_splitters import RecursiveCharacterTextSplitter
    #   확인: 나눈 뒤에도 각 청크의 metadata["page"]가 남아 있어야 출처를 보여줄 수 있어요.
    raise NotImplementedError


if __name__ == "__main__":
    from rag.config import settings
    from rag.load import load_pdf

    chunks = split_docs(load_pdf(settings.pdf_path), settings.chunk_size, settings.chunk_overlap)
    print(f"청크 수: {len(chunks)}")
    print(chunks[2].metadata, chunks[2].page_content[:200])
