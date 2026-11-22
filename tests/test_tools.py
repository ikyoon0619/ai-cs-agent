# 도구 함수 자체 테스트. 재고/시승은 가상 데이터라 Claude, DB 없이 바로 돌아간다.
from datetime import timedelta

import pytest

from rag import tools


@pytest.fixture(autouse=True)
def clear_bookings():
    tools.BOOKINGS.clear()            # 테스트끼리 예약이 섞이지 않게 매번 비움
    yield
    tools.BOOKINGS.clear()


def tomorrow():
    return (tools.today_kst() + timedelta(days=1)).isoformat()


def test_tool_schema_has_enum():
    # @tool이 타입 힌트(Literal)를 JSON 스키마의 enum으로 바꿨는지 = Claude가 보는 설명서
    schema = tools.check_inventory.args_schema.model_json_schema()
    assert "하이브리드" in schema["properties"]["powertrain"]["enum"]
    assert schema["required"] == ["powertrain"]


def test_check_inventory_filters():
    text = tools.check_inventory.invoke({"powertrain": "하이브리드", "trim": "익스클루시브"})
    assert "하이브리드 익스클루시브 / 그레이: 2대" in text
    assert "프리미엄" not in text


def test_check_inventory_not_sold():
    text = tools.check_inventory.invoke({"powertrain": "2.5 터보", "trim": "프리미엄"})
    assert "판매하지 않습니다" in text


def test_slots_reject_past_date():
    with pytest.raises(Exception):
        tools.get_test_drive_slots.invoke({"branch": "강남점", "date": tools.today_kst().isoformat()})


def test_book_then_slot_is_gone():
    day = tomorrow()
    slots = tools.available_slots("강남점", tools.parse_date(day))
    assert slots, "가상 데이터상 빈 시간이 하나는 있어야 함"
    time = slots[0]

    text = tools.book_test_drive.invoke({"branch": "강남점", "date": day, "time": time, "customer_name": "홍길동"})
    assert "예약 완료" in text and "TD-" in text
    assert time not in tools.available_slots("강남점", tools.parse_date(day))

    # 같은 시간을 또 예약하면 실패
    with pytest.raises(ValueError):
        tools.book_test_drive.invoke({"branch": "강남점", "date": day, "time": time, "customer_name": "김철수"})
