"""step 3. 청크를 임베딩해 Chroma에 저장하고, 저장된 것은 다시 불러온다."""

from langchain_chroma import Chroma
from langchain_core.documents import Document


def get_embeddings():
    # TODO: VoyageAIEmbeddings(model=settings.embedding_model)을 돌려주세요.
    #   힌트: from langchain_voyageai import VoyageAIEmbeddings
    raise NotImplementedError


def build_store(chunks: list[Document]) -> Chroma:
    # TODO: Chroma.from_documents(...)로 만들고 settings.chroma_dir에 저장하세요.
    raise NotImplementedError


def load_store() -> Chroma:
    # TODO: 이미 저장된 chroma_dir을 다시 임베딩하지 않고 불러오세요.
    #   힌트: Chroma(persist_directory=..., embedding_function=...)
    raise NotImplementedError
