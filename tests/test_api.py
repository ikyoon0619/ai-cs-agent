# tests/test_api.py
# 서버를 띄우지 않고 API를 호출해보는 테스트. JUnit + MockMvc 와 같은 역할
from fastapi.testclient import TestClient

import api.main
from rag.answer import Answer

client = TestClient(api.main.app)   # 가짜 브라우저. 진짜 서버 없이 요청을 보낼 수 있음


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_chat_ok(monkeypatch):
    # ask()를 가짜로 바꿔치기 → Claude/Voyage/DB 호출 없이 금방 끝남
    def fake_ask(question):
        return Answer(answer=f"가짜 답: {question}", sources=["27"])

    monkeypatch.setattr(api.main, "ask", fake_ask)   # api.main 안의 ask 이름을 가짜로 교체

    res = client.post("/chat", json={"question": "연비?"})
    assert res.status_code == 200
    assert res.json() == {"answer": "가짜 답: 연비?", "sources": ["27"]}


def test_chat_empty_question():
    # 빈 질문은 ask()까지 가지도 않고 422로 거절되어야 함
    res = client.post("/chat", json={"question": ""})
    assert res.status_code == 422