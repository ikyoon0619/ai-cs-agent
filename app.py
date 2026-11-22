# app.py
# 브라우저 채팅 화면: 질문 입력 → 사용량 확인 → ask() → 답변 + 출처 표시
import streamlit as st

from rag.answer import ask
from rag.usage import try_consume

SESSION_LIMIT = 10   # 한 사람(브라우저 탭)이 한 번에 물을 수 있는 질문 수

st.set_page_config(page_title="쏘나타 카탈로그 상담 챗봇", page_icon="🚗")
st.title("쏘나타 카탈로그 상담 챗봇")
st.caption(
    "쏘나타 디 엣지 카탈로그(PDF)만 근거로 답하고, 근거 페이지를 함께 보여 주는 RAG 데모입니다. "
    "검색: pgvector + BM25 하이브리드(RRF) · 답변: Claude · 임베딩: Voyage"
)

# 대화 기록 보관함 (새로고침 전까지 유지)
# Streamlit은 입력할 때마다 이 파일을 위에서부터 "다시 실행"한다.
# 그냥 변수에 저장하면 매번 사라지므로 st.session_state에 저장해야 한다.
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user"/"assistant", "content": 글, "sources": [...]}]


def show_message(role, content, sources=None):
    # 말풍선 하나 그리기. role이 "user"면 사람 아이콘, "assistant"면 봇 아이콘
    with st.chat_message(role):
        st.write(content)
        if sources:
            st.caption("출처: " + ", ".join(f"p.{p}" for p in sources))


# 0. 예시 질문 버튼: 처음 온 사람이 뭘 물어야 할지 바로 알 수 있게
EXAMPLES = ["트림 종류 알려줘", "연비가 어떻게 돼?", "RSPA가 뭐야?", "아반떼 연비는 얼마야?"]
clicked = None
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        clicked = example

# 1. 지난 대화를 처음부터 다시 그리기 (다시 실행될 때마다 화면이 비므로)
for msg in st.session_state.messages:
    show_message(msg["role"], msg["content"], msg.get("sources"))

# 2. 입력창. 사용자가 엔터를 치면 question에 글이 들어오고, 아니면 None
typed = st.chat_input("질문을 입력하세요", max_chars=500)
question = typed or clicked

if question:
    asked = sum(1 for m in st.session_state.messages if m["role"] == "user")
    if asked >= SESSION_LIMIT:
        st.warning(f"데모라서 한 번에 {SESSION_LIMIT}개까지만 질문할 수 있어요. 새로고침하면 다시 시작합니다.")
        st.stop()                       # 아래 코드는 실행하지 않고 여기서 끝
    if not try_consume():               # 오늘 전체 한도(DAILY_LIMIT) 확인 + 1 증가
        st.warning("오늘 데모 사용량을 모두 썼습니다. 내일 다시 시도해 주세요.")
        st.stop()

    # 3. 사용자 질문: 화면에 그리고 기록에 저장
    show_message("user", question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 4. 답 만들기 (기다리는 동안 빙글빙글 표시)
    try:
        with st.spinner("카탈로그를 찾는 중..."):
            result = ask(question)      # Answer 객체: result.answer, result.sources
    except Exception:
        st.error("답변을 만들지 못했어요. 잠시 후 다시 시도해 주세요.")
        st.stop()

    # 5. 봇 답변: 화면에 그리고 기록에 저장
    show_message("assistant", result.answer, result.sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )