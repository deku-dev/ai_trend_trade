import json, time

from openai import OpenAI, RateLimitError
from typing import Optional, List, Dict, Any
from loguru import logger

from config import OPENAI_API_KEY
from utils import load_features

MODEL = 'gpt-4o-mini'

clientGpt = OpenAI(api_key=OPENAI_API_KEY)

def analyze_with_gpt(ticker, data_5m, data_1d, fundamental_data):
    features = load_features()
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
    features = load_features()
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