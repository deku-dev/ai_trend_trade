import requests, time

from datetime import datetime, timedelta, timezone
from typing import Optional
from loguru import logger

from config import POLYGON_API_KEY

MAX_RETRIES = 3
INITIAL_BACKOFF = 1

def fetch_financial_prompt(
    ticker: str,
    days: int = 30,
    filing_date_to: Optional[str] = None
) -> str:
    """
    Повертає компактний рядок для ШІ з усіма метриками останнього квартального звіту:
    "ticker|end_date|metric1:value|metric2:value|..."
    """
    # Обчислення діапазону дат
    try:
        end_dt = datetime.fromisoformat(filing_date_to).replace(tzinfo=timezone.utc) if filing_date_to else datetime.now(timezone.utc)
    except ValueError:
        logger.error(f'Невірний формат дати: {filing_date_to}')
        return ''
    start_dt = end_dt - timedelta(days=days)

    params = {
        'ticker': ticker,
        'timeframe': 'quarterly',
        'limit': 1,
        'apiKey': POLYGON_API_KEY,
        'sort': 'filing_date',
        'order': 'desc',
        'filing_date.gte': start_dt.date().isoformat(),
        'filing_date.lte': end_dt.date().isoformat(),
    }
    # Запит з retry
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get('https://api.polygon.io/vX/reference/financials', params=params)
        if resp.status_code == 200:
            break
        if resp.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
        else:
            logger.error(f'HTTP {resp.status_code} при запиті: {resp.text}')
            return ''
    else:
        logger.error('Не вдалося отримати дані після повторних спроб')
        return ''

    # Парсинг
    try:
        item = resp.json().get('results', [])[0]
    except Exception:
        return ''

    end_date = item.get('end_date', '')
    financials = item.get('financials', {})

    # Збір всіх метрик
    metrics = []
    for section, fields in financials.items():
        for key, info in fields.items():
            val = info.get('value')
            if val is not None:
                # скорочення ключа до останньої частини
                short = key.split('_')[-1]
                metrics.append(f"{short}:{val}")

    # Форматування
    parts = [ticker, end_date] + metrics
    return '|'.join(parts)