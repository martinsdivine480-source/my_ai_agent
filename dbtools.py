"""Read-only MySQL access for the agent.

Needs:  pip install pymysql
Set these in .env (use a database account that only has SELECT permission):
  DB_HOST=127.0.0.1
  DB_PORT=3306
  DB_USER=readonly_user
  DB_PASSWORD=your_password
  DB_NAME=school_db
The tools are switched on only when DB_NAME is set."""
import json
import os
import re

MAX_ROWS = 200
MAX_RESULT_CHARS = 20000
MAX_SQL_CHARS = 2000
MAX_CELL_CHARS = 500
TIMEOUT_SECONDS = 10

ALLOWED_FIRST_WORDS = {"select", "show", "describe", "desc", "explain", "with"}
BLOCKED = re.compile(
    r"\binto\s+(?:outfile|dumpfile)\b|\bload_file\s*\(|\bsleep\s*\(|\bbenchmark\s*\("
    r"|\bfor\s+update\b|\block\s+in\s+share\s+mode\b",
    re.I,
)


def enabled() -> bool:
    return bool(os.getenv("DB_NAME", "").strip())


def check_sql(sql: str) -> str:
    """Accept only one plain read statement. Raises ValueError otherwise."""
    text = (sql or "").strip()
    if not text:
        raise ValueError("The SQL is empty.")
    if len(text) > MAX_SQL_CHARS:
        raise ValueError(f"The SQL is too long (limit is {MAX_SQL_CHARS} characters).")
    text = text.rstrip(";").strip()
    if ";" in text:
        raise ValueError("Only one statement is allowed, and it must not contain semicolons.")
    if "/*" in text:
        raise ValueError("Comments of the form /* */ are not allowed.")
    first = text.split(None, 1)[0].lower()
    if first not in ALLOWED_FIRST_WORDS:
        raise ValueError("Only read statements are allowed (SELECT, SHOW, DESCRIBE, EXPLAIN).")
    if BLOCKED.search(text):
        raise ValueError("That statement uses something that is not allowed for read-only access.")
    return text


def _connect():
    if not enabled():
        raise RuntimeError("The database is not set up. Add DB_HOST, DB_USER, DB_PASSWORD and "
                           "DB_NAME to the .env file.")
    try:
        import pymysql
    except ImportError:
        raise RuntimeError("The pymysql package is not installed. Run: pip install pymysql")
    conn = pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306") or 3306),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME"),
        charset="utf8mb4",
        connect_timeout=5,
        read_timeout=TIMEOUT_SECONDS,
        write_timeout=TIMEOUT_SECONDS,
        autocommit=True,
    )
    with conn.cursor() as cur:
        cur.execute("SET SESSION TRANSACTION READ ONLY")
        for setting in ("SET SESSION MAX_EXECUTION_TIME=5000",   # MySQL
                        "SET SESSION max_statement_time=5"):     # MariaDB (XAMPP)
            try:
                cur.execute(setting)
            except Exception:
                pass  # this server does not know that setting
    return conn


def _cell(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", "replace")
    text = str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "..."


def schema() -> dict:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name, column_name, column_type FROM information_schema.columns "
                "WHERE table_schema = DATABASE() ORDER BY table_name, ordinal_position"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    tables: dict = {}
    for table, column, column_type in rows:
        if table not in tables and len(tables) >= 200:
            continue
        columns = tables.setdefault(str(table), [])
        if len(columns) < 80:
            columns.append(f"{column} {column_type}")
    return {"database": os.getenv("DB_NAME"), "table_count": len(tables), "tables": tables}


def query(sql: str) -> dict:
    text = check_sql(sql)
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(text)
            fetched = cur.fetchmany(MAX_ROWS + 1)
            columns = [d[0] for d in (cur.description or [])]
    finally:
        conn.close()
    truncated = len(fetched) > MAX_ROWS
    rows = [[_cell(v) for v in row] for row in fetched[:MAX_ROWS]]
    while len(rows) > 1 and len(json.dumps(rows, default=str, ensure_ascii=False)) > MAX_RESULT_CHARS:
        rows = rows[: len(rows) // 2]
        truncated = True
    return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated}