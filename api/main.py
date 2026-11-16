# api/main.py
# FastAPI 서버: HTTP 요청을 받아 RAG 챗봇(ask)을 부른다
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.answer import ask, Answer   # 1주차에 만든 함수와 응답 모양을 그대로 재사용

app = FastAPI(title="쏘나타 카탈로그 상담 API")

logger = logging.getLogger("uvicorn.error")   # uvicorn 터미널에 찍히는 로거

class ChatRequest(BaseModel):
    # 요청 JSON의 모양. Spring의 @RequestBody DTO + @Valid 검증과 같다
    question: str = Field(min_length=1, max_length=500, description="고객 질문")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=Answer)   # 응답 모양 = 1주차의 Answer(answer, sources)
def chat(req: ChatRequest):
    # req: FastAPI가 요청 JSON을 ChatRequest로 자동 변환 + 검증해서 넣어줌
    try:
        return ask(req.question)            # 검색 → Claude → Answer 객체
    except Exception as e:
        logger.exception("chat 실패")   # 서버 터미널에 전체 에러(traceback) 기록
        raise HTTPException(status_code=503, detail=f"답변 생성 실패: {type(e).__name__}")