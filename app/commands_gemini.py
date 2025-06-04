import json

from loguru import logger
from datetime import datetime
from typing import Dict, List
from telegram.constants import ParseMode
from telegram import Update
from telegram.ext import (
    ContextTypes
)

from app.utils_ai import validate_date, add_to_history, update_history, load_tickers
from app.chart_data import fetch_market_prompt
from app.financial_data import fetch_financial_prompt
from app.gemini_handler import analyze_with_gemini, analyze_multiple_with_gemini

async def analyze_gem(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    if len(args) < 1 or len(args) > 2:
        await update.message.reply_text("Будь ласка, використовуйте /analyze <TICKER> [YYYY-MM-DD]")
        return

    ticker = args[0].upper()

    # Визначаємо дату аналізу
    target_date = datetime.now()
    if len(args) == 2:
        dt = validate_date(args[1])
        if not dt:
            await update.message.reply_text("Невірний формат дати. Використовуйте YYYY-MM-DD.")
            return
        target_date = dt

    await update.message.reply_text(
        f"Починаю аналіз тикера {ticker} за дату {target_date.strftime('%Y-%m-%d')}..."
    )
    try:
        data_5m = fetch_market_prompt(
            ticker, 5, 'minute', 5,
            end_date=target_date.strftime('%Y-%m-%d')
        )
        data_1d = fetch_market_prompt(
            ticker, 1, 'day', 30,
            end_date=target_date.strftime('%Y-%m-%d')
        )
        fundamental_data = fetch_financial_prompt(ticker, filing_date_to=target_date.strftime('%Y-%m-%d'))

        result = analyze_with_gemini(ticker, data_5m, data_1d, fundamental_data, user_id=update.effective_user.id)
        add_to_history(ticker)
        update_history(ticker, result)

        pretty = json.dumps(result, ensure_ascii=False, indent=2)
        await update.message.reply_text(f"<pre>{pretty}</pre>", parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        await update.message.reply_text(f"Помилка під час аналізу {ticker}.")

async def analyze_all_gem(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args[:]  # копія списка аргументів
    # 1) Визначаємо дату, якщо передано останнім аргументом
    target_date = datetime.now()
    ticker_args: List[str] = []
    if args:
        dt = validate_date(args[-1])
        if dt:
            target_date = dt
            args = args[:-1]
        # Якщо після видалення дати є інші args — це тикери
        if args:
            # розбиваємо усі передані токени за комами
            for token in args:
                ticker_args += [t.strip().upper() for t in token.split(',') if t.strip()]

    # 2) Якщо тикери не передали — завантажуємо усі з CSV
    if ticker_args:
        tickers = ticker_args
    else:
        tickers = load_tickers()

    # 3) Сповіщаємо користувача про початок
    await update.message.reply_text(
        f"Починаю аналіз тикерів на {target_date.strftime('%Y-%m-%d')}: {', '.join(tickers)}"
    )

    # 4) Збираємо дані для кожного тикера
    data_5m_map: Dict[str, str] = {}
    data_1d_map: Dict[str, str] = {}
    fundamental_map: Dict[str, str] = {}

    for tk in tickers:
        try:
            data_5m_map[tk] = fetch_market_prompt(
                tk, multiplier=5, timespan='minute', days=3,
                end_date=target_date.strftime('%Y-%m-%d')
            )
            data_1d_map[tk] = fetch_market_prompt(
                tk, multiplier=1, timespan='day', days=30,
                end_date=target_date.strftime('%Y-%m-%d')
            )
            fundamental_map[tk] = fetch_financial_prompt(
                tk, filing_date_to=target_date.strftime('%Y-%m-%d')
            )
            # Додаємо до історії (результат буде оновлено пізніше)
            add_to_history(tk)
        except Exception as e:
            logger.error(f"Error fetching data for {tk}: {e}")
            # Якщо не вдалося отримати дані для цього тикера — виключаємо його
            tickers.remove(tk)

    # 5) Аналіз усіх одразу й ранжування
    try:
        ranked_results = analyze_multiple_with_gemini(
            tickers,
            data_5m_map,
            data_1d_map,
            fundamental_map,
            user_id=update.effective_user.id
        )
    except Exception as e:
        logger.error(f"Error in multi-ticker analysis: {e}")
        await update.message.reply_text("Не вдалося виконати масовий аналіз. Спробуйте пізніше.")
        return

    # 6) Надсилаємо відсортований результат та оновлюємо історію
    for item in ranked_results:
        rec = item
        # якщо отримали "string" замість dict — спробуємо розпарсити
        if isinstance(rec, str):
            try:
                rec = json.loads(rec)
            except json.JSONDecodeError:
                logger.warning(f"Cannot parse record: {rec}")
                continue

        # пропускаємо некоректні формати
        if not isinstance(rec, dict) or "ticker" not in rec:
            logger.warning(f"Skipping unexpected record format: {rec!r}")
            continue

        tk = rec["ticker"]
        # оновлюємо результат в історії
        try:
            update_history(tk, rec)
        except Exception:
            logger.warning(f"Не вдалося оновити історію для {tk}")

        # надсилаємо користувачу
        pretty = json.dumps(rec, ensure_ascii=False, indent=2)
        await update.message.reply_text(
            f"<b>{tk}</b>\n<pre>{pretty}</pre>",
            parse_mode=ParseMode.HTML
        )