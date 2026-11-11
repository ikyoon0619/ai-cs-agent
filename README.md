# ai-cs-agent

문서 기반 RAG로 고객 질문에 답하고 도구를 호출하는 AI CS 에이전트입니다.
자동차 카탈로그 PDF 챗봇에서 시작해 단계별로 확장하고 있습니다.

## 진행 상황

- [x] step 1: 카탈로그 PDF 로드 (PyPDFLoader, 29페이지)
- [x] step 2: 청크 분할
- [ ] step 3: 임베딩 + 벡터 DB 저장
- [ ] step 4: 검색
- [ ] step 5: 출처와 함께 답변

## 실행 방법

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # API 키 입력
# data/ 폴더에 카탈로그 PDF를 넣은 뒤
python -m rag.load
```

## 메모

- `langchain-community`는 지원 종료 예정이라 대체 로더를 검토할 예정입니다.
