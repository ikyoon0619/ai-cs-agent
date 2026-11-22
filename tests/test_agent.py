# 에이전트 반복문 테스트. 진짜 Claude 대신 "정해진 순서로 답하는 가짜 LLM"을 넣는다.
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from rag import agent, tools


class FakeLLM:
    """invoke할 때마다 준비된 AIMessage를 하나씩 돌려준다. 받은 메시지도 기록해 둔다."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_agent_calls_tool_then_answers(monkeypatch):
    fake = FakeLLM([
        AIMessage("", tool_calls=[tool_call("check_inventory", {"powertrain": "LPG"}, "c1")]),
        AIMessage("LPG 프리미엄 화이트가 2대 있습니다. (가상 데이터)"),
    ])
    monkeypatch.setattr(agent, "get_agent_llm", lambda: fake)

    result = agent.run_agent("LPG 재고 있어?")

    assert result.tools_used == ["check_inventory"]
    assert "2대" in result.answer
    # 두 번째 Claude 호출에는 도구 결과(ToolMessage)가 붙어 있어야 한다
    last = fake.calls[1][-1]
    assert isinstance(last, ToolMessage) and last.tool_call_id == "c1"
    assert "LPG 프리미엄 / 화이트: 2대" in last.content


def test_agent_sources_only_from_retrieved_pages(monkeypatch):
    class FakeSearcher:
        def search(self, query, k=3):
            return [Document(page_content="트렁크 480ℓ", metadata={"page_label": "27"})]

    monkeypatch.setattr(tools, "get_searcher", lambda: FakeSearcher())
    fake = FakeLLM([
        AIMessage("", tool_calls=[tool_call("search_catalog", {"query": "트렁크 용량"}, "c1")]),
        AIMessage("트렁크는 480ℓ입니다 (p.27). 참고로 p.99도 보세요."),   # p.99는 검색 안 된 페이지
    ])
    monkeypatch.setattr(agent, "get_agent_llm", lambda: fake)

    result = agent.run_agent("트렁크 용량은?")
    assert result.sources == ["27"]          # 지어낸 p.99는 출처에서 빠짐


def test_agent_tool_error_goes_back_to_claude(monkeypatch):
    fake = FakeLLM([
        AIMessage("", tool_calls=[tool_call("get_test_drive_slots", {"branch": "강남점", "date": "어제"}, "c1")]),
        AIMessage("날짜를 YYYY-MM-DD로 알려 주세요."),
    ])
    monkeypatch.setattr(agent, "get_agent_llm", lambda: fake)

    result = agent.run_agent("어제 시승")
    error = fake.calls[1][-1]
    assert error.status == "error" and "날짜 형식" in error.content   # 서버는 안 죽고 Claude가 에러를 읽음
    assert result.answer.startswith("날짜를")


def test_agent_stops_after_max_steps(monkeypatch):
    # 매번 새 메시지 객체를 만든다 (LangGraph는 같은 id의 메시지를 "교체"하므로, 진짜 Claude처럼 매번 달라야 함)
    fake = FakeLLM([AIMessage("", tool_calls=[tool_call("check_inventory", {"powertrain": "LPG"}, f"c{i}")])
                    for i in range(agent.MAX_STEPS)])
    monkeypatch.setattr(agent, "get_agent_llm", lambda: fake)

    result = agent.run_agent("계속 도구만 부르는 경우")
    assert len(fake.calls) == agent.MAX_STEPS        # Claude 호출이 상한에서 멈춤
    assert "단계가 너무 많이" in result.answer
    assert result.tools_used == ["check_inventory"] * (agent.MAX_STEPS - 1)   # 마지막 요청은 실행 안 됨


def test_history_is_trimmed():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(30)]
    messages = agent.build_messages("질문", history)
    # system 1 + 최근 이력 MAX_HISTORY개 + 이번 질문 1
    assert len(messages) == 1 + agent.MAX_HISTORY + 1
    assert messages[-2].content == "m29"


def test_graph_shape():
    # 그래프 모양: START → agent → (tools ↔ agent) → END
    from rag.graph import build_graph
    g = build_graph(FakeLLM([])).get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    assert ("__start__", "agent") in edges
    assert ("agent", "tools") in edges and ("tools", "agent") in edges
    assert ("agent", "__end__") in edges


def test_agent_parallel_tool_calls(monkeypatch):
    # Claude가 한 번에 도구 2개를 부르면 둘 다 실행되고, 결과 2개가 순서대로 붙는다
    fake = FakeLLM([
        AIMessage("", tool_calls=[tool_call("check_inventory", {"powertrain": "LPG"}, "a"),
                                  tool_call("check_inventory", {"powertrain": "하이브리드"}, "b")]),
        AIMessage("둘 다 있습니다."),
    ])
    monkeypatch.setattr(agent, "get_agent_llm", lambda: fake)

    result = agent.run_agent("LPG랑 하이브리드 재고")
    assert result.tools_used == ["check_inventory", "check_inventory"]
    assert [m.tool_call_id for m in fake.calls[1][-2:]] == ["a", "b"]


# ---------------------------------------------------------------------------
# 상담원 승인 (human-in-the-loop)
# ---------------------------------------------------------------------------
def booking_args():
    from datetime import timedelta
    day = tools.today_kst() + timedelta(days=1)
    time = tools.available_slots("강남점", day)[0]
    return {"branch": "강남점", "date": day.isoformat(), "time": time, "customer_name": "홍길동"}


def test_booking_pauses_then_approved(monkeypatch):
    tools.BOOKINGS.clear()
    args = booking_args()
    fake = FakeLLM([
        AIMessage("", tool_calls=[tool_call("book_test_drive", args, "b1")]),
        AIMessage("예약이 확정됐습니다. (가상 데이터)"),
    ])
    monkeypatch.setattr(agent, "get_agent_llm", lambda: fake)

    # ① 첫 실행: 예약 직전에 멈춤. 아직 예약 안 됨
    first = agent.run_agent("내일 강남점 예약해줘")
    assert first.pending is not None and first.pending.tool == "book_test_drive"
    assert first.pending.args == args
    assert tools.BOOKINGS == {}
    assert len(fake.calls) == 1                     # Claude는 한 번만 불림

    # ② 상담원 승인 → 멈춘 곳부터 이어서 실행 → 예약 1건
    second = agent.resume_agent(first.pending.thread_id, approved=True)
    assert second.pending is None
    assert second.tools_used == ["book_test_drive"]
    assert len(tools.BOOKINGS) == 1                 # 다시 실행돼도 예약은 딱 한 번
    assert "예약 완료" in fake.calls[1][-1].content

    # ③ 끝난 요청은 다시 승인할 수 없음 (메모리에서도 지워짐)
    import pytest
    with pytest.raises(LookupError):
        agent.resume_agent(first.pending.thread_id, approved=True)
    tools.BOOKINGS.clear()


def test_booking_rejected(monkeypatch):
    tools.BOOKINGS.clear()
    fake = FakeLLM([
        AIMessage("", tool_calls=[tool_call("book_test_drive", booking_args(), "b1")]),
        AIMessage("죄송합니다. 다른 시간을 골라 주세요."),
    ])
    monkeypatch.setattr(agent, "get_agent_llm", lambda: fake)

    first = agent.run_agent("예약해줘")
    second = agent.resume_agent(first.pending.thread_id, approved=False)
    assert tools.BOOKINGS == {}                     # 거절하면 예약 안 됨
    rejected = fake.calls[1][-1]
    assert rejected.status == "error" and "승인하지 않았습니다" in rejected.content
    assert second.answer.startswith("죄송합니다")


def test_invalid_booking_skips_approval(monkeypatch):
    # 지난 날짜처럼 어차피 안 되는 예약은 상담원에게 묻지 않고 바로 Claude에게 에러를 돌려준다
    bad = {"branch": "강남점", "date": "2000-01-01", "time": "10:00", "customer_name": "홍길동"}
    fake = FakeLLM([
        AIMessage("", tool_calls=[tool_call("book_test_drive", bad, "b1")]),
        AIMessage("내일부터 예약할 수 있어요."),
    ])
    monkeypatch.setattr(agent, "get_agent_llm", lambda: fake)

    result = agent.run_agent("2000년 1월 1일 예약")
    assert result.pending is None
    assert fake.calls[1][-1].status == "error"


def test_history_replays_tool_calls():
    # 이전 답이 도구로 만든 답이면, 그때의 도구 호출 → 결과 → 답을 그대로 다시 보여 줘야 한다
    # (글만 넘기면 Claude가 "아까 재고는 조회 안 하고 답했다"고 잘못 사과함)
    history = [{"role": "user", "content": "하이브리드 재고 있어?"},
               {"role": "assistant", "content": "4대 있습니다.",
                "tools": [{"name": "check_inventory", "args": {"powertrain": "하이브리드"}, "result": "재고 4대"},
                          {"name": "없는도구", "args": {}, "result": "x"}]}]
    messages = agent.build_messages("시승 가능해?", history)
    # system, user, assistant(도구 요청), tool(결과), assistant(답), user(이번 질문)
    kinds = [type(m).__name__ for m in messages]
    assert kinds == ["SystemMessage", "HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"]
    assert messages[2].tool_calls[0]["name"] == "check_inventory"      # 모르는 도구는 버림
    assert messages[3].content == "재고 4대" and messages[3].tool_call_id == messages[2].tool_calls[0]["id"]
    assert messages[4].content == "4대 있습니다."


def test_answer_steps_only_from_current_question(monkeypatch):
    # 다시 넣어 준 이전 대화의 도구 호출은 이번 답의 tools_used/steps에 섞이면 안 된다
    history = [{"role": "user", "content": "재고?"},
               {"role": "assistant", "content": "있어요",
                "tools": [{"name": "check_inventory", "args": {"powertrain": "LPG"}, "result": "2대"}]}]
    fake = FakeLLM([
        AIMessage("", tool_calls=[tool_call("check_inventory", {"powertrain": "하이브리드"}, "c1")]),
        AIMessage("하이브리드도 있어요."),
    ])
    monkeypatch.setattr(agent, "get_agent_llm", lambda: fake)

    result = agent.run_agent("하이브리드는?", history)
    assert result.tools_used == ["check_inventory"]
    assert len(result.steps) == 1 and result.steps[0].args == {"powertrain": "하이브리드"}
    assert "하이브리드" in result.steps[0].result
