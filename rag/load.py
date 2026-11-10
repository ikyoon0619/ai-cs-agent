# rag/load.py
# step 1: PDF 파일을 읽어서 "페이지 1장 = Document 1개" 리스트로 만든다.

from pathlib import Path  # 파일 경로를 문자열보다 안전하게 다루는 표준 라이브러리 (Java의 java.nio.file.Path와 비슷)

from langchain_community.document_loaders import PyPDFLoader  # LangChain이 제공하는 PDF 로더 (내부적으로 pypdf 사용)
from langchain_core.documents import Document  # 텍스트(page_content)와 부가정보(metadata)를 담는 LangChain의 기본 단위

# 읽을 PDF 위치. 저장소 맨 위 폴더에서 실행한다고 가정한 상대 경로예요.
PDF_PATH = Path("data/sonata-the-edge_catalog.pdf")


def load_pdf(path: Path) -> list[Document]:
    """PDF를 페이지 단위 Document 리스트로 돌려준다."""
    # 파일이 없으면 LangChain 내부의 알기 어려운 에러 대신, 우리가 정한 메시지로 바로 알려준다.
    if not path.exists():
        raise FileNotFoundError(f"PDF가 없습니다: {path}")

    loader = PyPDFLoader(str(path))  # 로더 객체 생성 (아직 읽지는 않음)
    return loader.load()             # 실제로 읽기. 페이지마다 Document 하나씩 담긴 리스트를 돌려줌


# 이 파일을 직접 실행했을 때만 아래가 돈다. (다른 파일에서 import할 때는 실행 안 됨)
# Java의 public static void main(String[] args)와 같은 역할이에요.
if __name__ == "__main__":
    docs = load_pdf(PDF_PATH)

    print("페이지 수:", len(docs))                           # 리스트 길이 = PDF 페이지 수
    print("첫 페이지 metadata:", docs[0].metadata)            # source(파일 경로), page(페이지 번호) 등이 들어 있음
    print("4번째 페이지 본문 앞 200자:", docs[3].page_content[:200])  # [:200] = 앞에서 200글자까지 자르기