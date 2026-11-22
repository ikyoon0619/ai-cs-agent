# app.py
# 브라우저 채팅 화면: 질문 입력 → 사용량 확인 → run_agent() → 답변 + 출처 + 사용한 도구 표시
# 시승 예약은 바로 되지 않고 "상담원 승인" 카드가 뜬다 → 승인/거절 → resume_agent()로 이어서 실행
import streamlit as st

from rag.agent import resume_agent, run_agent, tool_labels
from rag.usage import try_consume

SESSION_LIMIT = 10   # 한 사람(브라우저 탭)이 한 번에 물을 수 있는 질문 수

st.set_page_config(page_title="쏘나타 상담 에이전트", page_icon="🚗")
st.title("쏘나타 상담 에이전트")
st.caption(
    "Claude가 질문을 보고 도구를 골라 씁니다: 카탈로그 검색(RAG), 재고 조회, 시승 시간 조회, 시승 예약. "
    "차량 정보는 카탈로그(PDF)만 근거로 답하고 페이지를 보여 줍니다. 재고와 시승은 데모용 가상 데이터입니다. "
    "시승 예약은 상담원이 승인해야 확정됩니다 (데모에서는 직접 상담원 역할을 해 보세요). "
    "검색: pgvector + BM25 하이브리드(RRF) · 에이전트: Claude Tool Use + LangGraph · 임베딩: Voyage"
)

# 대화 기록 보관함 (새로고침 전까지 유지)
# Streamlit은 입력할 때마다 이 파일을 위에서부터 "다시 실행"한다.
# 그냥 변수에 저장하면 매번 사라지므로 st.session_state에 저장해야 한다.
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user"/"assistant", "content": 글, "sources": [...], "tools": [...]}]
if "pending" not in st.session_state:
    st.session_state.pending = None  # 승인 대기 중인 예약 (PendingApproval) 또는 None


def show_message(role, content, sources=None, tools=None):
    # 말풍선 하나 그리기. role이 "user"면 사람 아이콘, "assistant"면 봇 아이콘
    with st.chat_message(role):
        st.write(content)
        if tools:
            st.caption("🔧 사용한 도구: " + ", ".join(tools))
        if sources:
            st.caption("출처: " + ", ".join(f"p.{p}" for p in sources))


def save_answer(result):
    # 봇 답변을 기록에 저장하고, 승인 대기면 기억해 둔다. 그린 다음 st.rerun()으로 화면을 새로 그린다
    st.session_state.messages.append({"role": "assistant", "content": result.answer,
                                      "sources": result.sources, "tools": tool_labels(result.tools_used)})
    st.session_state.pending = result.pending


# 0. 예시 질문 버튼: 처음 온 사람이 뭘 물어야 할지 바로 알 수 있게
EXAMPLES = ["트림 종류 알려줘", "RSPA가 뭐야?", "하이브리드 재고 있어?", "내일 강남점 시승 가능해?"]
clicked = None
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        clicked = example

# 1. 지난 대화를 처음부터 다시 그리기 (다시 실행될 때마다 화면이 비므로)
for msg in st.session_state.messages:
    show_message(msg["role"], msg["content"], msg.get("sources"), msg.get("tools"))

# 1-1. 상담원 승인 카드: 예약 요청이 멈춰 있으면 승인/거절 버튼을 보여 준다
pending = st.session_state.pending
if pending:
    a = pending.args
    with st.container(border=True):
        st.markdown("**🧑‍💼 상담원 승인 대기: 시승 예약**")
        st.write(f"{a.get('customer_name')}님 · {a.get('branch')} · {a.get('date')} {a.get('time')}")
        st.caption("실제 서비스에서는 상담원 화면에 뜨는 카드예요. 데모에서는 직접 눌러 보세요.")
        approve_col, reject_col = st.columns(2)
        decision = None
        if approve_col.button("✅ 승인", use_container_width=True):
            decision = True
        if reject_col.button("❌ 거절", use_container_width=True):
            decision = False
    if decision is not None:
        try:
            with st.spinner("예약을 처리하는 중..."):
                result = resume_agent(pending.thread_id, decision)   # 멈춘 그래프를 이어서 실행
        except LookupError:
            # 서버가 재시작되어 대기 중이던 요청이 사라진 경우 등
            st.session_state.pending = None
            st.warning("이 예약 요청은 만료됐어요. 다시 요청해 주세요.")
            st.stop()
        except Exception:
            st.error("예약을 처리하지 못했어요. 잠시 후 다시 시도해 주세요.")
            st.stop()
        save_answer(result)
        st.rerun()

# 2. 입력창. 사용자가 엔터를 치면 question에 글이 들어오고, 아니면 None
#    승인 대기 중에는 새 질문을 막는다 (먼저 승인/거절)
typed = st.chat_input("질문을 입력하세요" if not pending else "먼저 위 예약 요청을 승인하거나 거절해 주세요",
                      max_chars=500, disabled=bool(pending))
question = None if pending else (typed or clicked)

if question:
    asked = sum(1 for m in st.session_state.messages if m["role"] == "user")
    if asked >= SESSION_LIMIT:
        st.warning(f"데모라서 한 번에 {SESSION_LIMIT}개까지만 질문할 수 있어요. 새로고침하면 다시 시작합니다.")
        st.stop()                       # 아래 코드는 실행하지 않고 여기서 끝
    if not try_consume():               # 오늘 전체 한도(DAILY_LIMIT) 확인 + 1 증가
        st.warning("오늘 데모 사용량을 모두 썼습니다. 내일 다시 시도해 주세요.")
        st.stop()

    # 3. 사용자 질문: 화면에 그리고 기록에 저장
    #    에이전트에게 넘길 이전 대화는 질문을 저장하기 "전"의 기록이다
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    show_message("user", question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 4. 답 만들기 (기다리는 동안 빙글빙글 표시)
    try:
        with st.spinner("상담원이 확인하는 중..."):
            result = run_agent(question, history)   # AgentAnswer: answer, sources, tools_used, pending
    except Exception:
        st.error("답변을 만들지 못했어요. 잠시 후 다시 시도해 주세요.")
        st.stop()

    # 5. 봇 답변: 기록에 저장하고 화면을 다시 그린다 (승인 대기면 위에 승인 카드가 뜸)
    save_answer(result)
    st.rerun()