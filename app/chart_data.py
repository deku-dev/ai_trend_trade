import time, requests, random

from dateutil import parser
from loguru import logger
from typing import Optional, List
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, time as dtime

from config import POLYGON_API_KEY

MAX_RETRIES = 3
INITIAL_BACKOFF = 1
# Зона часу Нью-Йорку
NY_TZ = ZoneInfo("America/New_York")
# Стандартний час для end_date, якщо передано лише дату
DEFAULT_END_TIME = dtime(hour=9, minute=45)

def _parse_date(end_date: Optional[str]) -> datetime:
    """
    Парсить рядок end_date. Якщо передано лише дату, додає час 09:45 за Нью-Йорком.
    Якщо рядок включає час, використовує його (локалізує в Нью-Йорк, якщо час без часової зони).
    """
    if end_date is None:
        now_ny = datetime.now(NY_TZ)
        # встановлюємо сьогоднішній день із часом 09:45 New York
        return now_ny.replace(hour=DEFAULT_END_TIME.hour,
                              minute=DEFAULT_END_TIME.minute,
                              second=0, microsecond=0)
    try:
        dt = parser.isoparse(end_date)
    except (ValueError, TypeError):
        logger.error(f"Невірний формат дати: {end_date}")
        raise
    # Якщо час не вказано (тільки дата без 'T')
    if 'T' not in end_date:
        dt = dt.replace(hour=DEFAULT_END_TIME.hour,
                        minute=DEFAULT_END_TIME.minute)
    # Локалізуємо час до New York, якщо без tzinfo
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY_TZ)
    else:
        dt = dt.astimezone(NY_TZ)
    return dt

def _get_aggregates(ticker: str, multiplier: int, timespan: str,
                    start: str, end: str) -> List[dict]:
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {"apiKey": POLYGON_API_KEY}
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(url, params=params, timeout=10)
        logger.debug(f"Запит до Polygon API: {resp.url}")
        if resp.status_code == 200:
            return resp.json().get("results", [])
        elif resp.status_code == 429:
            sleep_time = backoff + random.uniform(0, backoff * 0.1)
            logger.warning(f"Rate limit hit, sleeping {sleep_time:.1f}s (attempt {attempt})")
            time.sleep(sleep_time)
            backoff *= 2
        else:
            resp.raise_for_status()

    raise RuntimeError(f"Не вдалося отримати дані за {MAX_RETRIES} спроб")


def fetch_market_prompt(
    ticker: str,
    multiplier: int,
    timespan: str,
    days: int,
    end_date: Optional[str] = None
) -> str:
    """
    Повертає компактні рядки для кожної свічки у форматі:
    "timestamp|o:open|h:high|l:low|c:close|v:volume|<індикатори>" кожен у новому рядку.

    end_date може бути ISO-строкою з датою або датою+часом. Якщо передано лише дату,
    час автоматично встановлюється на 09:45 за Нью-Йорком.
    Всі часи виводяться в Нью-Йоркській часовій зоні.
    """
    if days < 0:
        raise ValueError("Аргумент 'days' має бути невід’ємним")

    end_dt = _parse_date(end_date)
    start_dt = end_dt - timedelta(days=days)
    # Для API передаємо тільки дати в форматі YYYY-MM-DD
    s, e = start_dt.date().isoformat(), int(end_dt.timestamp())

    data = _get_aggregates(ticker, multiplier, timespan, s, e)
    if not data:
        logger.info("Отримано порожній список даних")
        return ""

    lines = []
    fields = ("o", "h", "l", "c", "v")
    indicators = ("DI+", "DI-", "ADX")

    for item in data:
        # Конвертуємо timestamp (мс UTC) в New York
        utc_dt = datetime.fromtimestamp(item["t"] / 1_000, tz=ZoneInfo("UTC"))
        ny_dt = utc_dt.astimezone(NY_TZ)
        # Форматуємо з врахуванням зміщення
        ts_str = ny_dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        parts = [ts_str]

        for key in fields:
            parts.append(f"{key}:{item.get(key, '')}")

        # Додаємо індикатори, якщо є
        for ind in indicators:
            if ind in item:
                parts.append(f"{ind}:{item[ind]}")

        lines.append("|".join(parts))

    return "\n".join(lines)