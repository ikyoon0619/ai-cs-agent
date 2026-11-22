# rag/hybrid.py
# 하이브리드 검색: 벡터 검색(뜻) + BM25 키워드 검색(단어)을 RRF로 합친다
import re

from rank_bm25 import BM25Okapi

from rag.store import get_store, load_chunks

def tokenize(text):
    """BM25용 단어 쪼개기.
    - 영어·숫자 덩어리(RSPA, t9m, e-dtvc, 12.6)는 통째로 한 단어
    - 한글은 2글자씩 겹쳐 자름: "트렁크가" → ["트렁", "렁크", "크가"]
      (형태소 분석기 없이도 '트렁크'와 '트렁크가'가 같은 조각을 공유해서 매칭됨)
    """
    tokens = []
    for word in re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*|[가-힣]+", text.lower()):
        if "가" <= word[0] <= "힣" and len(word) > 1:
            tokens += [word[i:i + 2] for i in range(len(word) - 1)]
        else:
            tokens.append(word)
    return tokens


class HybridSearcher:
    def __init__(self):
        self.store = get_store()                             # 벡터 검색용 (PostgreSQL)
        self.chunks = load_chunks()                          # BM25용 청크 44개 (DB에서 꺼냄, 임베딩 비용 0)
        if not self.chunks:
            raise RuntimeError("DB에 청크가 없습니다. 먼저 python -m scripts.ingest 를 실행하세요")      # BM25용 청크 44개 (임베딩 비용 0)
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in self.chunks])

    def keyword_search(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))       # 청크마다 BM25 점수
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in order[:k] if scores[i] > 0]   # 겹치는 단어가 없으면 제외

    def search(self, query, k=3, candidates=10, rrf_k=60):
        vector_docs = self.store.similarity_search(query, k=candidates)
        keyword_docs = self.keyword_search(query, candidates)

        scores = {}      # 청크 본문 → RRF 점수
        docs = {}        # 청크 본문 → Document
        for ranked in (vector_docs, keyword_docs):
            for rank, doc in enumerate(ranked):              # rank: 0, 1, 2, ...
                key = doc.page_content                       # 같은 청크면 본문이 같음
                docs[key] = doc
                scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)

        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [docs[key] for key in best]


if __name__ == "__main__":
    searcher = HybridSearcher()
    print(tokenize("RSPA가 뭐야? 트렁크 용량은 480ℓ"))
    for doc in searcher.search("RSPA가 뭐야?", k=3):
        print(doc.metadata["page_label"], doc.page_content[:60])