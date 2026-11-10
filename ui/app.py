"""Streamlit 채팅 화면. 실행: streamlit run ui/app.py"""

import streamlit as st

st.title("카탈로그 상담 챗봇")

question = st.chat_input("궁금한 점을 물어보세요")
if question:
    st.chat_message("user").write(question)
    # TODO (토요일): rag.answer.answer_question(question)을 호출해
    #   답변과 "출처: 12, 15페이지", confidence를 보여주세요.
    st.chat_message("assistant").write("아직 연결 전입니다.")
