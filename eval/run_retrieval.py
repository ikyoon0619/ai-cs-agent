# eval/run_retrieval.py
# 평가셋으로 검색 품질을 잰다. LLM 호출 없음
#   python -m eval.run_retrieval vector   (기본)
#   python -m eval.run_retrieval hybrid
import json
import sys

from rag.store import get_store

K = 5


def load_questions(path="eval/questions.jsonl"):
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def has_answer(doc, answers):
    text = " ".join(doc.page_content.split())
    return any(a in text for a in answers)


def get_search(mode):
    """mode에 따라 '질문, k → 청크 목록' 함수를 돌려준다"""
    if mode == "hybrid":
        from rag.hybrid import HybridSearcher
        searcher = HybridSearcher()
        return lambda question, k: searcher.search(question, k=k)
    store = get_store()
    return lambda question, k: store.similarity_search(question, k=k)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "vector"
    search = get_search(mode)
    print(f"=== 모드: {mode} ===")

    hit1 = hit3 = hit5 = 0
    rr_sum = 0.0          # MRR 계산용: 정답 순위의 역수 합
    total = 0

    for q in load_questions():
        if not q["pages"]:
            continue
        total += 1

        docs = search(q["question"], K)
        hits = [has_answer(d, q["answers"]) for d in docs]
        rank = hits.index(True) + 1 if any(hits) else None   # 정답 청크 순위 (없으면 None)

        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        hit5 += rank is not None
        rr_sum += 1 / rank if rank else 0

        pages = [d.metadata["page_label"] for d in docs]
        mark = "O" if rank and rank <= 3 else "X"
        print(f'{mark} [{q["type"]}] {q["question"]}  정답순위={rank or "-"}  검색페이지={pages}')

    print()
    print(f"Hit@1: {hit1}/{total} = {hit1 / total:.2f}")
    print(f"Hit@3: {hit3}/{total} = {hit3 / total:.2f}")
    print(f"Hit@5: {hit5}/{total} = {hit5 / total:.2f}")
    print(f"MRR  : {rr_sum / total:.3f}")


if __name__ == "__main__":
    main()