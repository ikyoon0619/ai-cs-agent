# rag/answer.py
# 질문 → 검색 → 청크를 글로 합치기 → 프롬프트에 끼우기 → Claude → 출처 붙은 답

from langchain_core.prompts import ChatPromptTemplate      # 빈칸 있는 질문 양식
from langchain_core.output_parsers import StrOutputParser  # Claude 답에서 글자만 꺼내는 도구

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


# 질문 양식 (편지 양식처럼 {context}, {question} 자리가 비어 있다)
prompt = ChatPromptTemplate.from_messages([
    # system = Claude에게 주는 역할과 규칙. 사용자에게는 안 보인다
    ("system",
     "너는 쏘나타 카탈로그 상담원이다. "
     "아래 카탈로그 내용만 근거로 답하고, 근거 페이지를 (p.번호)로 표시해라. "
     "근거가 없으면 '카탈로그에서 찾을 수 없습니다'라고 답해라."),
    # human = 실제로 Claude에게 보내는 메시지. 빈칸은 invoke할 때 채워진다
    ("human", "카탈로그 내용:\n{context}\n\n질문: {question}"),
])

# 체인: 부품 3개를 파이프(|)로 연결. 왼쪽 결과가 오른쪽으로 넘어간다
#   prompt          : 빈칸을 채워 완성된 메시지를 만든다
#   get_llm()       : 완성된 메시지를 Claude에게 보내고 답(AIMessage)을 받는다
#   StrOutputParser : 답에서 .content(글자)만 꺼낸다 → 결과가 그냥 문자열
chain = prompt | get_llm() | StrOutputParser()


def ask(question):
    # ① 검색: 질문과 뜻이 가까운 청크 3개
    docs = get_store().similarity_search(question, k=3)
    # ② 청크 3개 → "[p.27] ...\n\n[p.15] ..." 글 한 덩어리
    context = format_docs(docs)
    # ③ 양식의 빈칸 이름과 딕셔너리 키가 같아야 한다 ({context} ↔ "context")
    return chain.invoke({"context": context, "question": question})


if __name__ == "__main__":
    print(ask("기름 얼마나 먹어?"))