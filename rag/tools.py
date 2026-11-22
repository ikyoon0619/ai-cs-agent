# rag/tools.py
# Claude가 "골라서 부를 수 있는" 함수(도구) 4개.
#   - search_catalog       : 카탈로그 검색 (기존 하이브리드 RAG를 도구로 감쌈)
#   - check_inventory      : 재고 조회 (가상 데이터)
#   - get_test_drive_slots : 시승 가능 시간 조회 (가상 데이터)
#   - book_test_drive      : 시승 예약 (가상, 서버 메모리에만 저장)
#
# @tool 데코레이터가 하는 일
#   함수 이름 + docstring + 타입 힌트 → Claude에게 보내는 "도구 설명서(JSON 스키마)"로 변환.
#   Claude는 이 설명서만 보고 "어떤 도구를, 어떤 인자로" 부를지 정한다.
#   그래서 docstring은 사람용 주석이 아니라 Claude용 사용 설명서다.
#
# 재고/시승 데이터는 전부 가상(mock)이다. 실제 대리점 시스템과 연결되어 있지 않다.
import hashlib
import uuid
from datetime import date, datetime, timedelta
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from rag.answer import format_docs, get_searcher

KST = ZoneInfo("Asia/Seoul")

# ---------------------------------------------------------------------------
# 가상 데이터
# ---------------------------------------------------------------------------
Powertrain = Literal["2.0 가솔린", "1.6 터보", "2.5 터보", "하이브리드", "LPG"]
Trim = Literal["프리미엄", "익스클루시브", "인스퍼레이션", "N Line"]
Color = Literal["화이트", "블랙", "그레이", "블루"]
Branch = Literal["강남점", "판교점", "부산점"]

# (파워트레인, 트림) → {색상: 재고 대수}. 표에 없는 조합은 "판매하지 않는 조합"으로 본다.
INVENTORY = {
    ("2.0 가솔린", "프리미엄"):      {"화이트": 5, "블랙": 3, "그레이": 0, "블루": 1},
    ("2.0 가솔린", "익스클루시브"):  {"화이트": 2, "블랙": 0, "그레이": 4, "블루": 0},
    ("2.0 가솔린", "인스퍼레이션"):  {"화이트": 1, "블랙": 2, "그레이": 0, "블루": 0},
    ("1.6 터보", "프리미엄"):        {"화이트": 3, "블랙": 1, "그레이": 2, "블루": 0},
    ("1.6 터보", "익스클루시브"):    {"화이트": 0, "블랙": 2, "그레이": 1, "블루": 1},
    ("1.6 터보", "인스퍼레이션"):    {"화이트": 1, "블랙": 0, "그레이": 0, "블루": 0},
    ("1.6 터보", "N Line"):          {"화이트": 2, "블랙": 1, "그레이": 0, "블루": 3},
    ("2.5 터보", "N Line"):          {"화이트": 0, "블랙": 1, "그레이": 0, "블루": 2},
    ("하이브리드", "프리미엄"):      {"화이트": 4, "블랙": 2, "그레이": 1, "블루": 0},
    ("하이브리드", "익스클루시브"):  {"화이트": 0, "블랙": 0, "그레이": 2, "블루": 0},
    ("하이브리드", "인스퍼레이션"):  {"화이트": 1, "블랙": 1, "그레이": 0, "블루": 0},
    ("LPG", "프리미엄"):             {"화이트": 2, "블랙": 0, "그레이": 0, "블루": 0},
}

BRANCH_INFO = {
    "강남점": "서울 강남구 (가상 주소)",
    "판교점": "경기 성남시 분당구 (가상 주소)",
    "부산점": "부산 해운대구 (가상 주소)",
}
SLOT_TIMES = ["10:00", "11:00", "13:00", "14:00", "15:00", "16:00", "17:00"]
MAX_DAYS_AHEAD = 30            # 30일 뒤까지만 예약 가능

# 예약 저장소: 서버 메모리(dict). 서버가 재시작되면 사라진다 (데모용).
# 실제 서비스라면 DB 테이블에 저장해야 한다.
BOOKINGS = {}                  # booking_id → {"branch", "date", "time", "name"}


def today_kst():
    return datetime.now(KST).date()


def parse_date(text):
    """'2026-11-28' → date. 형식이 틀리거나 예약할 수 없는 날이면 ValueError."""
    try:
        day = date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"날짜 형식이 잘못됐습니다: {text!r}. YYYY-MM-DD 형식으로 주세요.")
    today = today_kst()
    if day <= today:
        raise ValueError("시승 예약은 내일부터 가능합니다.")
    if day > today + timedelta(days=MAX_DAYS_AHEAD):
        raise ValueError(f"시승 예약은 {MAX_DAYS_AHEAD}일 이내만 가능합니다.")
    return day


def is_pre_booked(branch, day, time):
    """가상 데이터: 이미 다른 손님이 잡아 둔 시간인지.
    지점+날짜+시간을 해시해서 정하므로 같은 입력이면 항상 같은 결과 (테스트하기 쉬움).
    파이썬 hash()는 실행할 때마다 값이 바뀌어서 hashlib을 쓴다."""
    key = f"{branch}|{day.isoformat()}|{time}".encode()
    return hashlib.sha256(key).digest()[0] % 3 == 0      # 약 1/3이 이미 찬 시간


def available_slots(branch, day):
    taken = {b["time"] for b in BOOKINGS.values() if b["branch"] == branch and b["date"] == day}
    return [t for t in SLOT_TIMES if t not in taken and not is_pre_booked(branch, day, t)]


def validate_booking(branch, date, time, customer_name):
    """예약 가능한지 검사. 안 되면 ValueError.
    상담원에게 승인을 묻기 "전"에도 부른다 → 어차피 안 되는 예약으로 상담원을 귀찮게 하지 않음."""
    if branch not in BRANCH_INFO:
        raise ValueError(f"없는 지점입니다: {branch}")
    day = parse_date(date)
    name = customer_name.strip()
    if not name or len(name) > 20:
        raise ValueError("예약자 이름은 1~20자로 주세요.")
    if time not in available_slots(branch, day):
        raise ValueError(f"{branch} {day} {time} 은(는) 예약할 수 없는 시간입니다. 가능한 시간을 먼저 조회하세요.")
    return day, name


# ---------------------------------------------------------------------------
# 도구 (Claude가 부르는 함수)
# ---------------------------------------------------------------------------
@tool(response_format="content_and_artifact")
def search_catalog(query: str) -> tuple[str, list]:
    """쏘나타 디 엣지 카탈로그(PDF)에서 관련 내용을 검색한다.
    사양, 트림 구성, 옵션, 연비, 제원, 안전 기능, 보증 등 차량 정보 질문에 사용한다.
    결과의 [p.N]은 카탈로그 페이지 번호다.

    Args:
        query: 검색할 내용. 고객 질문을 그대로 넣거나 핵심 단어로 바꿔서 넣는다.
    """
    docs = get_searcher().search(query, k=3)
    if not docs:
        return "검색 결과가 없습니다.", []
    # content(글)는 Claude가 읽고, artifact(Document 목록)는 Claude에게 안 보내고 코드에서만 쓴다
    # → 나중에 출처 페이지를 정확히 뽑을 때 사용
    return format_docs(docs), docs


@tool
def check_inventory(powertrain: Powertrain,
                    trim: Optional[Trim] = None,
                    color: Optional[Color] = None) -> str:
    """대리점 재고(즉시 출고 가능 대수)를 조회한다. 가상 데이터다.
    트림이나 색상을 모르면 비워 두면 해당 파워트레인의 전체 재고를 보여 준다.

    Args:
        powertrain: 파워트레인(엔진) 종류
        trim: 트림. 모르면 비운다
        color: 외장 색상. 모르면 비운다
    """
    lines = []
    for (pt, tr), colors in INVENTORY.items():
        if pt != powertrain or (trim and tr != trim):
            continue
        for c, qty in colors.items():
            if color and c != color:
                continue
            status = f"{qty}대 (즉시 출고)" if qty > 0 else "0대 (주문 후 약 4~6주)"
            lines.append(f"- {pt} {tr} / {c}: {status}")
    if not lines:
        return f"{powertrain} {trim or ''} 조합은 판매하지 않습니다.".replace("  ", " ")
    return "재고 현황 (가상 데이터)\n" + "\n".join(lines)


@tool
def get_test_drive_slots(branch: Branch, date: str) -> str:
    """특정 지점, 특정 날짜에 시승 예약이 가능한 시간을 조회한다. 가상 데이터다.

    Args:
        branch: 시승 지점
        date: 날짜 (YYYY-MM-DD). '내일', '이번 주 토요일' 같은 말은 오늘 날짜를 기준으로 바꿔서 넣는다
    """
    day = parse_date(date)
    slots = available_slots(branch, day)
    if not slots:
        return f"{branch} {day} 은(는) 예약 가능한 시간이 없습니다."
    return f"{branch} {day} 예약 가능 시간: " + ", ".join(slots)


@tool
def book_test_drive(branch: Branch, date: str, time: str, customer_name: str) -> str:
    """시승을 예약한다. 가상 예약이며 서버 메모리에만 저장된다.
    고객이 지점, 날짜, 시간, 이름을 모두 확인하고 예약하겠다고 말한 뒤에만 호출한다.
    호출하면 상담원 승인을 거친 뒤에 실제로 예약된다.

    Args:
        branch: 시승 지점
        date: 날짜 (YYYY-MM-DD)
        time: 시간 (HH:MM). get_test_drive_slots 에서 나온 시간 중 하나
        customer_name: 예약자 이름
    """
    day, name = validate_booking(branch, date, time, customer_name)
    booking_id = "TD-" + uuid.uuid4().hex[:6].upper()
    BOOKINGS[booking_id] = {"branch": branch, "date": day, "time": time, "name": name}
    return (f"예약 완료 (가상). 예약번호 {booking_id}: {name}님, "
            f"{branch}({BRANCH_INFO[branch]}) {day} {time}")


# 실행 전에 상담원 승인이 필요한 도구 (되돌리기 어려운 "행동"을 하는 도구)
NEEDS_APPROVAL = {"book_test_drive"}

TOOLS = [search_catalog, check_inventory, get_test_drive_slots, book_test_drive]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
TOOL_LABELS = {                       # 화면에 보여 줄 한글 이름
    "search_catalog": "카탈로그 검색",
    "check_inventory": "재고 조회",
    "get_test_drive_slots": "시승 시간 조회",
    "book_test_drive": "시승 예약",
}
