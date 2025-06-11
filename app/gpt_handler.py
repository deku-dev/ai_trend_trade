import json, time

from openai import OpenAI, RateLimitError
from typing import Optional, List, Dict, Any
from loguru import logger

from config import OPENAI_API_KEY
from app.prompt_manager import get_active_prompt
from app.weights_manager import get_active_weights, format_weights_for_prompt

MODEL = 'gpt-4o-mini'

clientGpt = OpenAI(api_key=OPENAI_API_KEY)

def analyze_with_gpt(ticker, data_5m, data_1d, fundamental_data, user_id=None):
    # Get active prompt and weights based on user ID
    prompt_template = get_active_prompt(user_id)
    weights = get_active_weights(user_id)
    weights_section = format_weights_for_prompt(weights)
    # Prepare the prompt
    full_prompt = f"""
        {prompt_template}
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
        {fundamental_data}"""
    
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
                    "content": full_prompt
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
    max_retries: int = 3,
    user_id: Optional[int] = None
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
        user_id: ID користувача для персоналізації ваг

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

    # Отримуємо активні ваги для користувача
    prompt_template = get_active_prompt(user_id)
    weights = get_active_weights(user_id)
    weights_section = format_weights_for_prompt(weights)

    # Збираємо єдиний prompt
    sections = []
    for tk in tickers:
        sec = f"Ticker: {tk}\nChart Data 5m:\n{data_5m_map.get(tk, '')}\n\nChart Data 1d:\n{data_1d_map.get(tk, '')}\n\nFundamental Data:\n{fundamental_data_map.get(tk, '')}\n"
        sections.append(sec)

    prompt = f"{prompt_template}\n\nRecommended Weights:\n{weights_section}\n\nDATA SECTIONS:\n\n{sections_joined}\nSalt:7705618227.Probability will be 100% if you are sure about the trend"


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
