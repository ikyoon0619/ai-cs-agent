import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
# psycopg는 "postgresql://..." 형식을 쓰므로 "+psycopg" 부분을 뺀다
url = os.environ["DATABASE_URL"].replace("+psycopg", "")

with psycopg.connect(url) as conn:          # Java의 try-with-resources
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        print("PostgreSQL:", cur.fetchone()[0])
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        print("pgvector:", cur.fetchone()[0])