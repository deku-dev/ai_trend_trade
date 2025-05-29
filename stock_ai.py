from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field
import requests
from openai import OpenAI, RateLimitError
from loguru import logger
import random
import json
import time
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo
from dateutil import parser
from google import genai

from utils import add_adx

from config import OPENAI_API_KEY, POLYGON_API_KEY, GEMINI_API

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

clientGpt = OpenAI(api_key=OPENAI_API_KEY)
clientGemini = genai.Client(api_key=GEMINI_API)

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
            response = clientGpt.chat.completions.create(
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


class IntradayTrendMovementProbability(BaseModel):
    probability_value: str = Field(
        ...,
        description="Оцінка ймовірності значного інтрадейного руху у відсотках, наприклад '75%'"
    )
    confidence: int = Field(
        ...,
        ge=1,
        le=10,
        description="Рівень впевненості від 1 до 10"
    )
    justification: str = Field(
        ...,
        description="Коротке пояснення з ключовими факторами"
    )
    fundamental_impact: str = Field(
        ...,
        description="Ключові слова щодо впливу фундаментальних даних"
    )
    extra: Optional[str] = Field(
        None,
        description="Додатковий короткий коментар або прогноз"
    )
    
class IntradayAnalysis(BaseModel):
    ticker: str = Field(
        ...,
        description="Тикер акції, наприклад 'AAPL'"
    )
    intraday_trend_movement_probability: IntradayTrendMovementProbability

def analyze_with_gemini(
    ticker: str,
    data_5m: str,
    data_1d: str,
    fundamental_data: str,
):
    """
    Виконує аналіз інтрадея для заданого тикера за допомогою Gemini-4.

    Параметри:
        ticker: Тікер акції, наприклад "AAPL".
        data_5m: Серіалізовані дані 5-хвилинного графіка.
        data_1d: Серіалізовані дані денного графіка.
        fundamental_data: Текстові дані фундаментального аналізу.
        features_path: Шлях до JSON-файлу з рекомендованими вагами ознак.
        model: Інстанс моделі Gemini з методом generate_content.
        max_retries: Кількість спроб запиту, якщо щось піде не так.
        retry_delay: Затримка між спробами (у секундах).

    Повертає:
        Словник із полями:
            {
                "ticker": str,
                "intraday_trend_movement_probability": {
                    "probability_value": str,
                    "confidence": str,
                    "justification": str,
                    "fundamental_impact": str,
                    "extra": str
                }
            }
        або рядок з повідомленням про помилку.
    """
    # Завантажуємо ваги ознак
    features = load_features(FEATURES_PATH)
    weights_section = "\n".join(f"- {k}: weight {v}" for k, v in features.items())

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
        Chart Data 5m:
        {data_5m}
        Chart Data 1d:
        {data_1d}
        Fundamental Data:
        {fundamental_data}
        """.strip()
        
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = clientGemini.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": list[IntradayAnalysis],
                },
            )
            text = response.text.strip()
            analysis = json.loads(text)
            return analysis

        except json.JSONDecodeError:
            logger.warning(
                f"[{ticker}] Невалідний JSON на спробі {attempt}/{max_retries}: {text}"
            )
        except Exception as e:
            logger.error(f"[{ticker}] Помилка на спробі {attempt}/{max_retries}: {e}", exc_info=True)

        if attempt < max_retries:
            time.sleep(1000)

    error_msg = f"Аналіз {ticker} не виконано після {max_retries} спроб."
    logger.error(error_msg)
    return error_msg

def analyze_multiple_with_gpt(
    tickers: List[str],
    data_5m_map: Dict[str, str],
    data_1d_map: Dict[str, str],
    fundamental_data_map: Dict[str, str],
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """
    Аналізує одночасно декілька тикерів, ранжує їх за
    ймовірністю суттєвого intraday-трендового руху.

    Параметри:
        tickers: список рядків із символами акцій.
        data_5m_map: словник ticker -> текст із 5-хв свічками.
        data_1d_map: словник ticker -> текст із 1-д свічками.
        fundamental_data_map: словник ticker -> текст із фундаменталкою.
        max_retries: кількість повторних спроб при помилці.

    Повертає:
        Список об’єктів виду
        [
          {
            "ticker": "string",
            "probability_value": int,
            "confidence": int (1-10),
            "justification": "...",
            "fundamental_impact": "...",
            "extra": "..."
          },
          ...
        ]
        вже відсортований за probability_value DESC.
    """

    # Підвантажити ваги з FEATURES_PATH
    features = load_features(FEATURES_PATH)
    weights_section = "\n".join([f"- {k}: weight {v}" for k, v in features.items()])

    # Збираємо єдиний prompt
    sections = []
    for tk in tickers:
        sec = (
            f"Ticker: {tk}\n"
            f"Chart Data 5m:\n{data_5m_map.get(tk, '')}\n\n"
            f"Chart Data 1d:\n{data_1d_map.get(tk, '')}\n\n"
            f"Fundamental Data:\n{fundamental_data_map.get(tk, '')}\n"
        )
        sections.append(sec)

    prompt = (
        "You are an intraday stock analyst. Given multiple tickers below, analyze each using only 5-min and 1-day charts, "
        "ADX, DI+, DI– values, and a light evaluation of fundamentals. Then RANK the tickers by the probability (%) of a "
        "significant intraday trend movement today after the open (100 = most likely, 0 = least likely).\n\n"
        "Analysis for each must include:\n"
        "• Trend (price action, MA 5m/1d)\n"
        "• Momentum (ADX, DI+, DI–)\n"
        "• Volume\n"
        "• Fundamentals (low weight)\n\n"
        "Requirements:\n"
        "• For each ticker: probability_value (integer %), confidence (1–10), justification (short keywords), fundamental_impact, extra.\n"
        "• Finally, return a JSON array sorted by probability_value descending.\n\n"
        "Recommended Weights:\n"
        f"{weights_section}\n\n"
        "DATA SECTIONS:\n\n"
        f"{'---\n'.join(sections)}"
    )

    # Підготовка повідомлень
    messages = [
        {
            "role": "system",
            "content": "You are a stock market analyst with a high level of expertise in predicting trend movements."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    attempts = max_retries
    while attempts > 0:
        try:
            resp = clientGpt.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content
            # Парсимо JSON
            result = json.loads(raw)
            # Переконаємося, що це список, і він вже відсортований
            if isinstance(result, list):
                return result
            # Якщо модель повернула обʼєкт з полем "ranked", спробуємо витягти
            if isinstance(result, dict) and "ranked" in result:
                return result["ranked"]
            # Інакше — віддаємо так, як є
            return result

        except RateLimitError:
            logger.warning("Rate limit exceeded, retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error in multi-ticker analysis: {e}")
        attempts -= 1

    if raw is None:
        raise RuntimeError("Multi-ticker analysis failed after retries")
    
    # Парсимо JSON
    parsed = raw if isinstance(raw, (list, dict)) else json.loads(raw)

    # Нормалізуємо до списку словників
    result_list: List[Dict[str, Any]] = []

    if isinstance(parsed, list):
        result_list = parsed

    elif isinstance(parsed, dict):
        # Якщо є ключ "ranked" або "results" з масивом
        for key in ("ranked", "results"):
            if key in parsed and isinstance(parsed[key], list):
                result_list = parsed[key]
                break

        # Якщо ще порожньо — йдемо по парах ticker->data
        if not result_list and all(isinstance(v, dict) for v in parsed.values()):
            for tk, data in parsed.items():
                entry = {"ticker": tk}
                entry.update(data)
                result_list.append(entry)

    else:
        raise ValueError("Unexpected format from GPT: must be list or dict")

    # Переконаємося, що всі entries мають необхідні поля
    cleaned: List[Dict[str, Any]] = []
    for rec in result_list:
        if not isinstance(rec, dict) or "ticker" not in rec:
            continue
        try:
            rec["probability_value"] = int(rec["probability_value"])
            rec["confidence"] = int(rec.get("confidence", 0))
            cleaned.append(rec)
        except Exception:
            continue

    # Сортуємо за спаданням probability_value
    cleaned.sort(key=lambda x: x["probability_value"], reverse=True)
    return cleaned

def analyze_multiple_with_gemini(
    tickers: List[str],
    data_5m_map: Dict[str, str],
    data_1d_map: Dict[str, str],
    fundamental_data_map: Dict[str, str],
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Аналізує одночасно декілька тикерів за допомогою Gemini.
    Ранжує їх за ймовірністю суттєвого intraday-трендового руху.

    Параметри:
        tickers: список символів акцій.
        data_5m_map: ticker -> текст із 5-хв свічками.
        data_1d_map: ticker -> текст із 1-д свічками.
        fundamental_data_map: ticker -> текст із фундаментального аналізу.
        max_retries: кількість спроб при помилці.

    Повертає:
        Відсортований за probability_value DESC список словників виду:
        {
          "ticker": str,
          "probability_value": int,
          "confidence": int,
          "justification": str,
          "fundamental_impact": str,
          "extra": str
        }
    """
    # Завантажуємо рекомендовані ваги ознак
    features = load_features(FEATURES_PATH)
    weights_section = "\n".join(f"- {k}: weight {v}" for k, v in features.items())

    # Формуємо секції даних для кожного тикера
    sections = []
    for tk in tickers:
        sections.append(
            f"Ticker: {tk}\n"
            f"Chart Data 5m:\n{data_5m_map.get(tk, '')}\n\n"
            f"Chart Data 1d:\n{data_1d_map.get(tk, '')}\n\n"
            f"Fundamental Data:\n{fundamental_data_map.get(tk, '')}\n"
        )
    data_block = "\n---\n".join(sections)

    # Формуємо промпт для Gemini
    prompt = f"""
        You are an intraday stock analyst with the seriousness and discipline of a professional trader. My grandfather is terminally ill and has always dreamed of, in his final days, understanding which single stock is most likely to break out and trend today. He placed his last hope in your analysis.

        Given the list of tickers below, analyze each using only:
        • 5-minute and 1-day price charts (include moving averages MA5, MA20 where relevant)
        • ADX, DI+, DI– for momentum strength
        • Volume spikes and divergences
        • Key support/resistance levels and imminent breakouts
        • Fundamentals with low weight

        For each ticker, determine:
        1. probability_value: integer % likelihood of a **significant intraday trend** today (after market open), factoring in clear breakout above resistance or breakdown below support  
        2. confidence: integer 1–10  
        3. justification: very concise keywords (“ADX>25, volume spike, broke R1”)  
        4. fundamental_impact: brief note on any fundamental driver  
        5. extra: optional short outlook (“watch RSI for pullback”)

        **Important:**  
        - Treat each ticker as if you were risking your own capital.  
        - Be ruthless and precise: only the strongest breakout setups should score near 100%.  
        - Return a **pure** JSON array, sorted by probability_value DESC.  

        Recommended Weights for Your Analysis:
        {weights_section}

        DATA SECTIONS:
        {data_block}
    """.strip()

    attempt = 0
    last_error = None

    while attempt < max_retries:
        attempt += 1
        try:
            resp = clientGemini.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                },
            )
            text = resp.text.strip()
            result = json.loads(text)
            # Переконаємося, що повернувся список словників
            if isinstance(result, list):
                # Сортування на всяк випадок
                result.sort(key=lambda x: int(x.get("probability_value", 0)), reverse=True)
                return result
            # Якщо модель загорнула масив у поле
            if isinstance(result, dict):
                for key in ("ranked", "results"):
                    if key in result and isinstance(result[key], list):
                        ranked = result[key]
                        ranked.sort(key=lambda x: int(x.get("probability_value", 0)), reverse=True)
                        return ranked
            # Інакше — повертаємо "as is"
            return result

        except json.JSONDecodeError as jde:
            last_error = f"JSON decode error on attempt {attempt}: {jde}"
            logger.warning(last_error)
        except Exception as e:
            last_error = f"Error on attempt {attempt}: {e}"
            logger.error(last_error, exc_info=True)

        # Затримка перед повторною спробою
        if attempt < max_retries:
            time.sleep(5)

    # Після max_retries — кидаємо або повертаємо помилку
    error_msg = f"Multi-ticker analysis failed after {max_retries} attempts: {last_error}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)