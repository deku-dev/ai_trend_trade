from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import requests
from openai import OpenAI, RateLimitError
from loguru import logger
import os
import random
import json
import time
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Optional, List
from zoneinfo import ZoneInfo
from dateutil import parser

from utils import add_adx

from config import OPENAI_API_KEY, POLYGON_API_KEY

load_dotenv()

DATA_PATH = 'data/tickers.csv'
OUTPUT_PATH = 'output/predictions.json'
OUTPUT_DIR = 'output/'
FEATURES_PATH = 'data/features.csv'
HISTORY_PATH = 'output/history.json'
MODEL = 'gpt-4o-mini'

FEATURES = pd.read_csv("data/features.csv").set_index("parameter")["weight"].to_dict()
TICKERS = pd.read_csv("data/tickers.csv")["ticker"].tolist()

MAX_RETRIES = 3
INITIAL_BACKOFF = 1

client = OpenAI(api_key=OPENAI_API_KEY)
session = requests.Session()

logger.add("logs/app.log", format="{time} | {level} | {message}", rotation="10 MB")

# Завантаження історії
def load_history():
    with open(HISTORY_PATH, 'r') as f:
        return json.load(f)

# Збереження історії
def save_history(history):
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)

# Перевірка, чи вже є запис для конкретної дати
def is_already_processed(ticker):
    history = load_history()
    today = datetime.now().strftime('%Y-%m-%d')
    if ticker in history and today in history[ticker]:
        return True
    return False

# Додавання нового запису в історію (з результатом за замовчуванням None)
def add_to_history(ticker):
    history = load_history()
    today = datetime.now().strftime('%Y-%m-%d')
    
    if ticker not in history:
        history[ticker] = {}
    
    if today not in history[ticker]:
        history[ticker][today] = {"result": None}
    
    save_history(history)

# Оновлення результату для запису
def update_history(ticker, result):
    history = load_history()
    today = datetime.now().strftime('%Y-%m-%d')
    
    if ticker in history and today in history[ticker]:
        history[ticker][today]["result"] = result
        save_history(history)

def load_features(filepath):
    df = pd.read_csv(filepath)
    features = dict(zip(df['parameter'], df['weight']))
    return features

# Load tickers from CSV file
def load_tickers(filepath):
    return pd.read_csv(filepath)['ticker'].tolist()



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


# Analyze data with ChatGPT
def analyze_with_gpt(ticker, data_5m, data_1d, fundamental_data):
    features = load_features(FEATURES_PATH)
    weights_section = "\n".join([f"- {key}: weight {value}" for key, value in features.items()])
    
    prompt = f"""
        You are an intraday stock analyst. Analyze ticker "{ticker}" using only 5-min and 1-day charts, ADX, DI+, DI– values, and a light evaluation of fundamentals. Estimate the probability (%) of a significant intraday trend movement today after the open.
        Analysis must include:
        • Trend (price action, MA 5m/1d)
        • Momentum (ADX, DI+, DI–)
        • Volume
        • Fundamentals (low weight)
        Requirements:
        • Strict probability (%) for significant trend movement today
        • Confidence (1–10)
        • Justification: short, key words/conclusion (main factors)
        • Extra: optional brief remark or outlook
        Recommended Weights:
        {weights_section}
        Respond in this format:
        ```json
        {{
            "ticker": "{ticker}",
            "intraday_trend_movement_probability": {{
                "probability_value": "in %",
                "confidence": "int 1-10",
                "justification": "Short conclusion, key factors, keywords.",
                "fundamental_impact": "brief assessment, keywords",
                "extra": "Optional brief remark or outlook."
            }}
        }}
        ```
        Chart Data 5m:
        {data_5m}
        Chart Data 1d:
        {data_1d}
        Fundamental Data:
        {fundamental_data}
    """
    
    attempts = 3
    while attempts > 0:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "system",
                    "content": "You are a stock market analyst with a high level of expertise in predicting trend movements."
                }, {
                    "role": "user", 
                    "content": prompt
                }],
                response_format={"type": "json_object"}
            )
            
            analysis_text = response.choices[0].message.content
            usage = response.usage
            logger.info(f"Tokens used: input: {usage.prompt_tokens}, out: {usage.completion_tokens}, total: {usage.total_tokens}")
            analysis_json = json.loads(analysis_text)
            return analysis_json
        except RateLimitError:
            logger.error(f'Rate limit exceeded for GPT-4. Retrying analysis for {ticker} in 5 seconds...')
            time.sleep(5)
        except Exception as e:
            logger.error(f'Error during GPT analysis for {ticker}: {e}')
        attempts -= 1
    return 'Аналіз не виконано через помилки.'
