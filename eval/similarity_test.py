# eval/similarity_test.py
from rag.embed import get_embeddings

sentences = [
    "쏘나타 연비는 얼마야?",         # 0번: 기준 문장
    "기름 얼마나 먹어?",             # 1번: 단어는 다르지만 뜻이 비슷
    "에어백은 몇 개 있어?",          # 2번: 전혀 다른 주제
]

embeddings = get_embeddings()
vectors = embeddings.embed_documents(sentences)   # 3문장을 한 번에 변환 (요청 1번)


def similarity(a, b):
    # 같은 칸끼리 곱해서 전부 더한다 (클수록 뜻이 가깝다)
    score = 0
    for i in range(len(a)):
        score = score + a[i] * b[i]
    return score


print("0번 vs 1번 (뜻 비슷):", round(similarity(vectors[0], vectors[1]), 3))
print("0번 vs 2번 (뜻 다름):", round(similarity(vectors[0], vectors[2]), 3))