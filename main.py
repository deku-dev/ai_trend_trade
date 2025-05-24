from datetime import datetime, timedelta
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes
)
from config import TELEGRAM_API_KEY
from loguru import logger
import os, json

from stock_ai import load_tickers, fetch_market_prompt, analyze_with_gpt, add_to_history, update_history, fetch_financial_prompt, DATA_PATH, HISTORY_PATH, OUTPUT_DIR

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Ініціалізація файлу історії
if not os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'w') as f:
        json.dump({}, f)
        
def validate_date(date_str: str) -> datetime | None:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

        result = analyze_with_gpt(ticker, data_5m, data_1d, fundamental_data)
        # result = {}
        add_to_history(ticker)
        update_history(ticker, result)

        pretty = json.dumps(result, ensure_ascii=False, indent=2)
        await update.message.reply_text(f"<pre>{pretty}</pre>", parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        await update.message.reply_text(f"Помилка під час аналізу {ticker}.")

# Обробник команди /analyze_all
async def analyze_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    target_date = datetime.now()
    if len(args) == 1:
        dt = validate_date(args[0])
        if not dt:
            await update.message.reply_text("Невірний формат дати. Використовуйте YYYY-MM-DD.")
            return
        target_date = dt

    tickers = load_tickers(DATA_PATH)
    await update.message.reply_text(
        f"Починаю аналіз усіх тикерів за дату {target_date.strftime('%Y-%m-%d')}: {', '.join(tickers)}"
    )
    for ticker in tickers:
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

            result = analyze_with_gpt(ticker, data_5m, data_1d, fundamental_data)
            add_to_history(ticker)
            update_history(ticker, result)

            pretty = json.dumps(result, ensure_ascii=False, indent=2)
            await update.message.reply_text(
                f"<b>{ticker}</b>\n<pre>{pretty}</pre>",
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            await update.message.reply_text(f"Помилка під час аналізу {ticker}.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привіт! Я бот для аналізу акцій.\n"
        "/analyze <TICKER> [YYYY-MM-DD] - проаналізувати тикер за обрану дату (якщо не вказана, сьогодні)\n"
        "/analyze_all [YYYY-MM-DD] - проаналізувати всі тикери за дату (за замовчуванням сьогодні)"
    )

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_API_KEY).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CommandHandler("analyze_all", analyze_all))

    logger.info("Бот запущено.")
    app.run_polling()
