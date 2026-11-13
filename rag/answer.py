# rag/answer.py
# 질문 → 검색 → 청크를 글로 합치기 → 프롬프트에 끼우기 → Claude → 출처 붙은 답
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate      # 빈칸 있는 질문 양식

from rag.store import get_store   # 3일차: 저장된 Chroma 열기
from rag.llm import get_llm       # 4일차 #18: Claude 만들기


def format_docs(docs):
    # 청크(Document) 3개 → Claude가 읽을 수 있는 글 한 덩어리
    # 각 청크 앞에 [p.27]처럼 페이지를 붙여야 Claude가 답에 출처를 쓸 수 있다
    parts = []
    for doc in docs:
        page = doc.metadata["page_label"]              # 카탈로그에 찍힌 쪽 번호
        parts.append(f"[p.{page}] {doc.page_content}")  # 청크 본문
    return "\n\n".join(parts)                          # 청크 사이를 빈 줄로 구분



class Answer(BaseModel):
    # Claude가 채워야 하는 칸 2개. description은 Claude가 읽는 "칸 설명"
    answer: str = Field(description="카탈로그 근거로 쓴 답변")
    sources: list[str] = Field(description="근거로 쓴 페이지 번호 목록. 근거가 없으면 빈 리스트")


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "너는 쏘나타 카탈로그 상담원이다. "
     "아래 카탈로그 내용만 근거로 답해라. 근거로 쓴 페이지 번호는 sources에 넣어라. "
     "근거가 없으면 answer에 '카탈로그에서 찾을 수 없습니다'라고 쓰고 sources는 비워라."),
    ("human", "카탈로그 내용:\n{context}\n\n질문: {question}"),
])

chain = prompt | get_llm().with_structured_output(Answer, method="json_schema")   # 빈칸: "구조화된 출력과 함께"


def ask(question):
    # ① 검색: 질문과 뜻이 가까운 청크 3개
    docs = get_store().similarity_search(question, k=3)
    # ② 청크 3개 → "[p.27] ...\n\n[p.15] ..." 글 한 덩어리
    context = format_docs(docs)
    # ③ 양식의 빈칸 이름과 딕셔너리 키가 같아야 한다 ({context} ↔ "context")
    return chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    for question in ["트림 종류 알려줘", "아반떼 연비는 얼마야?"]:
        result = ask(question)
        print("==== 질문:", question)
        print("답변:", result.answer)
        print("출처:", result.sources)     # 리스트로 나와야 함
        print()