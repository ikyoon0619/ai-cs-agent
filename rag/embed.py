# rag/embed.py
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()   # .env 파일을 읽어서 VOYAGE_API_KEY를 환경변수로 등록


def get_embeddings():
    # 글자 → 좌표(숫자 1024개) 변환기를 만들어서 돌려준다
    return VoyageAIEmbeddings(model="voyage-3.5")      # 빈칸 ①: 클래스 이름 / 빈칸 ②: 모델 이름


if __name__ == "__main__":
    embeddings = get_embeddings()
    vector = embeddings.embed_query("쏘나타 연비는?")   # 빈칸 ③: 문장 하나를 변환하는 메서드

    print("벡터 길이:", len(vector))
    print("앞 5개 숫자:", vector[:5])