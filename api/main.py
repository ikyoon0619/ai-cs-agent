# api/main.py
# FastAPI 서버: HTTP 요청을 받아 상담 에이전트(run_agent)를 부른다
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.agent import AgentAnswer, Turn, run_agent   # Tool Use 에이전트 (검색 + 재고 + 시승)
from rag.usage import try_consume    # 하루 질문 수 제한

app = FastAPI(title="쏘나타 카탈로그 상담 API")

logger = logging.getLogger("uvicorn.error")   # uvicorn 터미널에 찍히는 로거

class ChatRequest(BaseModel):
    # 요청 JSON의 모양. Spring의 @RequestBody DTO + @Valid 검증과 같다
    question: str = Field(min_length=1, max_length=500, description="고객 질문")
    # 이전 대화 (선택). 시승 예약처럼 여러 번 주고받아야 하는 대화를 이어가려고 받는다
    history: list[Turn] = Field(default_factory=list, max_length=20, description="이전 대화")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=AgentAnswer)   # 응답 모양: answer, sources, tools_used
def chat(req: ChatRequest):
    # req: FastAPI가 요청 JSON을 ChatRequest로 자동 변환 + 검증해서 넣어줌
    if not try_consume():
        # 429 Too Many Requests: 오늘 한도를 다 씀. Claude를 부르지 않으니 비용 0
        raise HTTPException(status_code=429, detail="오늘 데모 사용량을 모두 썼습니다. 내일 다시 시도해 주세요.")
    try:
        return run_agent(req.question, req.history)   # Claude가 도구를 골라 쓰고 → AgentAnswer
    except Exception as e:
        logger.exception("chat 실패")   # 서버 터미널에 전체 에러(traceback) 기록
        raise HTTPException(status_code=503, detail=f"답변 생성 실패: {type(e).__name__}")
