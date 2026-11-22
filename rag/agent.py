# rag/agent.py
# Tool Use 에이전트: Claude가 질문을 보고 필요한 도구를 골라 부르고, 결과를 읽고, 최종 답을 쓴다.
# 반복 흐름(agent ↔ tools)은 rag/graph.py의 LangGraph 그래프가 맡고,
# 이 파일은 프롬프트·대화 이력 준비와 최종 답 정리(출처, 사용한 도구)를 맡는다.
#
# 핵심: Claude는 함수를 "직접" 실행하지 않는다.
#       "check_inventory를 이 인자로 불러 줘"라고 요청(tool_call)만 하고, 실행은 우리 코드가 한다.
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from rag.graph import MAX_STEPS, build_graph   # noqa: F401  (MAX_STEPS는 테스트에서 씀)
from rag.llm import get_llm
from rag.tools import KST, TOOL_LABELS, TOOLS

MAX_HISTORY = 10       # 이전 대화는 최근 10개 메시지만 보낸다 (토큰 비용 제한)
WEEKDAYS = "월화수목금토일"

SYSTEM_PROMPT = """너는 쏘나타 디 엣지 판매 상담원이다. 오늘은 {today}({weekday}요일)이다.

도구 사용 규칙:
- 차량 사양, 트림, 옵션, 연비, 제원, 보증 같은 질문은 search_catalog로 찾은 내용만 근거로 답한다.
  근거로 쓴 문장 끝에 (p.27)처럼 페이지를 붙인다.
  찾아도 근거가 없으면 '카탈로그에서 찾을 수 없습니다'라고 답한다. 추측하지 않는다.
- 재고나 출고 질문은 check_inventory를 쓴다.
- 시승은 get_test_drive_slots로 가능한 시간을 먼저 보여 준다.
  book_test_drive는 고객이 지점, 날짜, 시간, 이름을 확정하고 예약해 달라고 한 뒤에만 부른다.
  빠진 정보가 있으면 도구를 부르지 말고 고객에게 물어본다.
- 재고와 시승 정보는 데모용 가상 데이터라는 점을 답에 한 번 밝힌다.
- 쏘나타와 관계없는 질문은 정중히 거절한다.

답변은 한국어로, 3~6문장 정도로 짧게 쓴다."""


class Turn(BaseModel):
    # 이전 대화 한 줄. API 요청과 Streamlit 화면이 같은 모양을 쓴다
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=2000)


class AgentAnswer(BaseModel):
    answer: str = Field(description="최종 답변")
    sources: list[str] = Field(default_factory=list, description="답에 인용한 카탈로그 페이지")
    tools_used: list[str] = Field(default_factory=list, description="이번 답에 사용한 도구 이름 (순서대로)")


def get_agent_llm():
    # bind_tools: "이 도구들을 쓸 수 있어"라고 도구 설명서(JSON 스키마)를 요청에 붙인다.
    # 테스트에서는 이 함수를 가짜 LLM으로 바꿔치기한다.
    return get_llm().bind_tools(TOOLS)


def build_messages(question, history=()):
    now = datetime.now(KST)
    messages = [SystemMessage(SYSTEM_PROMPT.format(today=now.date().isoformat(),
                                                   weekday=WEEKDAYS[now.weekday()]))]
    for turn in list(history)[-MAX_HISTORY:]:
        role, content = (turn.role, turn.content) if isinstance(turn, Turn) else (turn["role"], turn["content"])
        messages.append(HumanMessage(content) if role == "user" else AIMessage(content))
    messages.append(HumanMessage(question))
    return messages


def cited_pages(answer, docs):
    """답에 적힌 (p.27) 중에서 실제로 검색된 페이지만 출처로 인정한다 (지어낸 페이지 걸러냄)."""
    retrieved = {str(d.metadata.get("page_label")) for d in docs}
    pages = []
    for page in re.findall(r"p\.\s*(\d+)", answer):
        if page in retrieved and page not in pages:
            pages.append(page)
    return pages


def run_agent(question, history=()):
    # 그래프 실행: 처음 상태(메시지 목록)를 넣으면 END까지 돌고 마지막 상태를 돌려준다
    graph = build_graph(get_agent_llm())
    state = graph.invoke({"messages": build_messages(question, history), "steps": 0})
    messages = state["messages"]

    # 그래프가 남긴 메시지에서 "사용한 도구"와 "검색된 청크"를 모은다
    tools_used, retrieved_docs = [], []
    for msg in messages:
        if isinstance(msg, AIMessage):
            tools_used += [call["name"] for call in msg.tool_calls]
        elif isinstance(msg, ToolMessage) and msg.name == "search_catalog" and msg.artifact:
            retrieved_docs += msg.artifact

    last = messages[-1]
    if last.tool_calls:                   # MAX_STEPS에서 끊겨서 도구 요청이 남은 채로 끝남
        return AgentAnswer(answer="질문을 처리하는 데 단계가 너무 많이 필요합니다. 질문을 나눠서 다시 물어봐 주세요.",
                           tools_used=tools_used[:-len(last.tool_calls)])   # 실행 안 된 요청은 빼기
    answer = last.text.strip() or "답변을 만들지 못했습니다."
    return AgentAnswer(answer=answer, sources=cited_pages(answer, retrieved_docs), tools_used=tools_used)


def tool_labels(names):
    """['search_catalog', 'search_catalog', 'check_inventory'] → ['카탈로그 검색', '재고 조회']"""
    labels = []
    for name in names:
        label = TOOL_LABELS.get(name, name)
        if label not in labels:
            labels.append(label)
    return labels


if __name__ == "__main__":
    for q in ["트림 종류 알려줘", "하이브리드 익스클루시브 재고 있어?", "내일 강남점 시승 가능한 시간 알려줘"]:
        result = run_agent(q)
        print("==== 질문:", q)
        print("도구:", result.tools_used)
        print("답변:", result.answer)
        print("출처:", result.sources)
        print()
