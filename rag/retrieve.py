"""step 4. 질문과 비슷한 청크를 점수와 함께 찾는다."""

from langchain_core.documents import Document


def search(question: str, k: int) -> list[tuple[Document, float]]:
    # TODO: load_store().similarity_search_with_score(question, k=k)
    raise NotImplementedError


if __name__ == "__main__":
    for doc, score in search("하이브리드 모델 연비는?", k=4):
        print(f"p.{doc.metadata['page'] + 1}  score={score:.3f}  {doc.page_content[:60]!r}")
