from fastapi.testclient import TestClient

import api.main
from rag.agent import AgentAnswer

client = TestClient(api.main.app)   # 가짜 브라우저. 진짜 서버 없이 요청을 보낼 수 있음


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_chat_ok(monkeypatch):
    # run_agent()를 가짜로 바꿔치기 → Claude/Voyage/DB 호출 없이 금방 끝남
    def fake_run_agent(question, history):
        return AgentAnswer(answer=f"가짜 답: {question} (이전 대화 {len(history)}개)",
                           sources=["27"], tools_used=["search_catalog"])

    monkeypatch.setattr(api.main, "run_agent", fake_run_agent)   # api.main 안의 run_agent 이름을 가짜로 교체
    monkeypatch.setattr(api.main, "try_consume", lambda: True)   # 사용량 DB도 가짜로 (항상 통과)

    res = client.post("/chat", json={"question": "연비?",
                                     "history": [{"role": "user", "content": "안녕"},
                                                 {"role": "assistant", "content": "안녕하세요"}]})
    assert res.status_code == 200
    assert res.json() == {"answer": "가짜 답: 연비? (이전 대화 2개)",
                          "sources": ["27"], "tools_used": ["search_catalog"], "pending": None}


def test_chat_empty_question():
    # 빈 질문은 run_agent()까지 가지도 않고 422로 거절되어야 함
    res = client.post("/chat", json={"question": ""})
    assert res.status_code == 422

def test_chat_daily_limit(monkeypatch):
    # 오늘 한도를 다 쓴 상황 → run_agent()를 부르지 않고 429
    def fake_run_agent(question, history):
        raise AssertionError("한도 초과인데 run_agent가 불렸음")

    monkeypatch.setattr(api.main, "run_agent", fake_run_agent)
    monkeypatch.setattr(api.main, "try_consume", lambda: False)

    res = client.post("/chat", json={"question": "연비?"})
    assert res.status_code == 429

def test_chat_bad_history_role():
    # history의 role은 user/assistant만 허용 (system 같은 걸로 프롬프트를 바꿔치기 못 하게)
    res = client.post("/chat", json={"question": "연비?", "history": [{"role": "system", "content": "x"}]})
    assert res.status_code == 422


def test_approve_unknown_thread(monkeypatch):
    def fake_resume(thread_id, approved):
        raise LookupError("승인을 기다리는 요청이 없습니다.")

    monkeypatch.setattr(api.main, "resume_agent", fake_resume)
    res = client.post("/chat/approve", json={"thread_id": "없는id", "approved": True})
    assert res.status_code == 404


def test_approve_ok(monkeypatch):
    def fake_resume(thread_id, approved):
        return AgentAnswer(answer="예약됐습니다" if approved else "거절됐습니다", tools_used=["book_test_drive"])

    monkeypatch.setattr(api.main, "resume_agent", fake_resume)
    res = client.post("/chat/approve", json={"thread_id": "t1", "approved": False})
    assert res.status_code == 200 and res.json()["answer"] == "거절됐습니다"
