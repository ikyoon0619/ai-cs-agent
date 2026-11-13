# app.py
# 브라우저 채팅 화면: 질문 입력 → ask() → 답변 + 출처 표시
import streamlit as st

from rag.answer import ask

st.title("쏘나타 카탈로그 상담 챗봇")

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


# 1. 지난 대화를 처음부터 다시 그리기 (다시 실행될 때마다 화면이 비므로)
for msg in st.session_state.messages:
    show_message(msg["role"], msg["content"], msg.get("sources"))

# 2. 입력창. 사용자가 엔터를 치면 question에 글이 들어오고, 아니면 None
question = st.chat_input("질문을 입력하세요")

if question:
    # 3. 사용자 질문: 화면에 그리고 기록에 저장
    show_message("user", question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 4. 답 만들기 (기다리는 동안 빙글빙글 표시)
    with st.spinner("카탈로그를 찾는 중..."):
        result = ask(question)          # Answer 객체: result.answer, result.sources

    # 5. 봇 답변: 화면에 그리고 기록에 저장
    show_message("assistant", result.answer, result.sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )