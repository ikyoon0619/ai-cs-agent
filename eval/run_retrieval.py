# eval/run_retrieval.py
# 평가셋으로 검색 품질을 잰다. LLM은 부르지 않음 (질문 임베딩 20번만)
#  - 페이지 Hit@5 : 정답 페이지가 상위 5개 안에 있나 (후한 기준, 참고용)
#  - 청크 Hit@3   : 정답 문구가 든 청크가 상위 3개 안에 있나 (챗봇이 실제로 Claude에 넘기는 개수)
#  - 청크 Hit@5   : 같은 기준, 상위 5개
import json

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
    # 청크 본문의 줄바꿈·공백을 한 칸으로 정리한 뒤, 정답 문구 중 하나라도 들어 있으면 True
    text = " ".join(doc.page_content.split())
    return any(a in text for a in answers)


def main():
    store = get_store()
    questions = load_questions()

    page5 = chunk3 = chunk5 = total = 0

    for q in questions:
        if not q["pages"]:                        # "없음" 문항은 제외
            continue
        total += 1

        docs = store.similarity_search(q["question"], k=K)
        found = [d.metadata["page_label"] for d in docs]
        hits = [has_answer(d, q["answers"]) for d in docs]   # 청크별 정답 포함 여부 [True, False, ...]

        page5 += any(p in q["pages"] for p in found)
        chunk3 += any(hits[:3])
        chunk5 += any(hits)

        # 정답 청크가 몇 등에 있었는지 (없으면 -)
        rank = hits.index(True) + 1 if any(hits) else "-"
        mark = "O" if any(hits[:3]) else "X"
        print(f'{mark} [{q["type"]}] {q["question"]}  정답청크순위={rank}  검색페이지={found}')

    print()
    print(f"페이지 Hit@5: {page5}/{total} = {page5 / total:.2f}  (참고)")
    print(f"청크  Hit@3: {chunk3}/{total} = {chunk3 / total:.2f}  ← 기준 지표")
    print(f"청크  Hit@5: {chunk5}/{total} = {chunk5 / total:.2f}")


if __name__ == "__main__":
    main()