# ai-cs-agent

문서 기반 RAG로 고객 질문에 답하고 도구를 호출하는 AI CS 에이전트입니다.
자동차 카탈로그 챗봇에서 시작해 5주 동안 단계별로 키워 나갑니다.

**스택:** Python 3.12 · LangChain · Claude · Voyage 임베딩 · Chroma → pgvector(2주차) · Streamlit · FastAPI(2주차) · LangGraph(4주차)

## 진행 상황

| 주차 | 내용 | 상태 |
| --- | --- | --- |
| 1 | PDF 로드 → 분할 → 임베딩 → 검색 → 구조화된 답변 + Streamlit | 진행 중 |
| 2 | FastAPI + pgvector로 이식, 평가셋 baseline | |
| 3 | 하이브리드 검색 + 리랭킹, 비교표 | |
| 4 | Tool Use → LangGraph 에이전트, 사람 승인 | |
| 5 | LLM-as-a-Judge 평가, 배포 | |

## 실행

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # ANTHROPIC_API_KEY, VOYAGE_API_KEY 입력
cp ~/path/to/catalog.pdf data/catalog.pdf

python -m rag.load              # step 1
python -m rag.split             # step 2
python -m rag.retrieve          # step 4 (step 3 저장 후)
python -m rag.answer            # step 5
streamlit run ui/app.py
python -m pytest
```

## 구조

```
rag/       load → split → store → retrieve → answer (step 1~5)
prompts/   프롬프트 (버전별 파일)
ui/        Streamlit 화면
eval/      테스트 질문과 결과 기록
data/      PDF (git에 올리지 않음)
```
