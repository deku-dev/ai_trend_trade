import json, time

from google import genai
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger

from config import GEMINI_API
from app.utils_ai import load_features


clientGemini = genai.Client(api_key=GEMINI_API)

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
    features = load_features()
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
    features = load_features()
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