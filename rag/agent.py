# rag/agent.py
# Tool Use 에이전트: Claude가 질문을 보고 필요한 도구를 골라 부르고, 결과를 읽고, 최종 답을 쓴다.
#
#   질문 ──▶ Claude ──(도구 호출 요청)──▶ 우리 코드가 도구 실행 ──(결과)──▶ Claude ──▶ ... ──▶ 최종 답
#                 ▲                                                          │
#                 └──────────────────── 반복 (최대 MAX_STEPS번) ─────────────┘
#
# 핵심: Claude는 함수를 "직접" 실행하지 않는다.
#       "check_inventory를 이 인자로 불러 줘"라고 요청(tool_call)만 하고, 실행은 우리 코드가 한다.
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from rag.llm import get_llm
from rag.tools import KST, TOOL_LABELS, TOOLS, TOOLS_BY_NAME

MAX_STEPS = 5          # Claude 호출 최대 횟수. 무한 반복과 비용 폭주를 막는 안전장치
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


def run_tool(call):
    """Claude가 요청한 도구 하나를 실행해서 ToolMessage로 돌려준다.
    도구에서 에러가 나도 서버를 멈추지 않고, 에러 내용을 Claude에게 보여 준다.
    → Claude가 "그 시간은 안 된대요, 다른 시간 고르세요"처럼 스스로 수습할 수 있다."""
    tool = TOOLS_BY_NAME.get(call["name"])
    if tool is None:
        return ToolMessage(f"없는 도구입니다: {call['name']}", tool_call_id=call["id"], status="error")
    try:
        # call 전체(name, args, id)를 넘기면 결과가 ToolMessage로 나온다 (tool_call_id 자동 연결)
        return tool.invoke(call)
    except Exception as e:
        return ToolMessage(f"도구 실행 실패: {e}", tool_call_id=call["id"], status="error")


def cited_pages(answer, docs):
    """답에 적힌 (p.27) 중에서 실제로 검색된 페이지만 출처로 인정한다 (지어낸 페이지 걸러냄)."""
    retrieved = {str(d.metadata.get("page_label")) for d in docs}
    pages = []
    for page in re.findall(r"p\.\s*(\d+)", answer):
        if page in retrieved and page not in pages:
            pages.append(page)
    return pages


def run_agent(question, history=()):
    llm = get_agent_llm()
    messages = build_messages(question, history)
    tools_used = []
    retrieved_docs = []

    for _ in range(MAX_STEPS):
        ai = llm.invoke(messages)
        messages.append(ai)               # Claude의 답(도구 호출 요청 포함)을 대화에 그대로 붙인다

        if not ai.tool_calls:             # 도구를 더 안 부르면 = 최종 답
            answer = ai.text.strip() or "답변을 만들지 못했습니다."
            return AgentAnswer(answer=answer,
                               sources=cited_pages(answer, retrieved_docs),
                               tools_used=tools_used)

        # Claude가 한 번에 여러 도구를 부를 수도 있다 (예: 재고 조회 + 카탈로그 검색).
        # 결과는 요청 순서대로 모두 붙여야 한다.
        for call in ai.tool_calls:
            result = run_tool(call)
            messages.append(result)
            tools_used.append(call["name"])
            if call["name"] == "search_catalog" and result.artifact:
                retrieved_docs.extend(result.artifact)

    # MAX_STEPS번 안에 끝나지 않음
    return AgentAnswer(answer="질문을 처리하는 데 단계가 너무 많이 필요합니다. 질문을 나눠서 다시 물어봐 주세요.",
                       tools_used=tools_used)


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
