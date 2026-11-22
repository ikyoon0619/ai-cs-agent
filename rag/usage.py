# rag/usage.py
# 사용량 제한: 하루 전체 질문 수를 DB에 세고, 한도를 넘으면 막는다 (유료 API 비용 보호)
import os
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "200"))   # 하루 최대 질문 수 (전체 사용자 합계)
KST = ZoneInfo("Asia/Seoul")                          # 서버가 해외에 있어도 '하루'는 한국 날짜 기준


@lru_cache
def get_engine():
    # DB 연결 묶음(커넥션 풀)을 한 번만 만들어 재사용
    return create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)


@lru_cache                  # 테이블 만들기는 서버가 켜진 뒤 처음 한 번만
def init_usage_table():
    with get_engine().begin() as conn:
        conn.execute(text("""
                          CREATE TABLE IF NOT EXISTS usage_daily (
                                                                     day   DATE PRIMARY KEY,
                                                                     count INTEGER NOT NULL
                          )
                          """))


def try_consume(limit=None):
    """오늘 횟수를 1 올린다. 한도 안이면 True, 이미 한도면 False.
    INSERT ... ON CONFLICT 한 문장으로 처리 → 동시에 여러 명이 물어도 숫자가 꼬이지 않는다."""
    limit = DAILY_LIMIT if limit is None else limit
    today = datetime.now(KST).date()
    init_usage_table()
    with get_engine().begin() as conn:
        row = conn.execute(text("""
                                INSERT INTO usage_daily (day, count) VALUES (:day, 1)
                                    ON CONFLICT (day) DO UPDATE SET count = usage_daily.count + 1
                                                         WHERE usage_daily.count < :limit
                                                             RETURNING count
                                """), {"day": today, "limit": limit}).first()
    return row is not None and row[0] <= limit   # 한도에 걸리면 UPDATE가 안 돼서 row가 None


if __name__ == "__main__":
    for i in range(4):
        print(i + 1, "번째:", try_consume(limit=3))   # True, True, True, False