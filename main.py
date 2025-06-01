from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes
)
from telegram import Update
from config import TELEGRAM_API_KEY
from loguru import logger

from app.commands_gpt import analyze_gpt, analyze_all_gpt
from app.commands_gemini import analyze_gem, analyze_all_gem

logger.add("logs/app.log", format="{time} | {level} | {message}", rotation="10 MB")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привіт! Я бот для аналізу акцій.\n"
        "/analyze_gem <TICKER> [YYYY-MM-DD] - Gemini проаналізувати тикер за обрану дату (якщо не вказана, сьогодні)\n"
        "/analyze_gpt <TICKER> [YYYY-MM-DD] - ChatGpt проаналізувати тикер за обрану дату (якщо не вказана, сьогодні)\n"
        "/analyze_all_gpt ticker,ticker,ticker [YYYY-MM-DD] - ChatGpt проаналізувати всі тикери за дату (за замовчуванням сьогодні)\n"
        "/analyze_all_gem ticker,ticker,ticker [YYYY-MM-DD] - Gemini проаналізувати всі тикери за дату (за замовчуванням сьогодні)"
    )

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_API_KEY).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze_gpt", analyze_gpt))
    app.add_handler(CommandHandler("analyze_gem", analyze_gem))
    app.add_handler(CommandHandler("analyze_all_gpt", analyze_all_gpt))
    app.add_handler(CommandHandler("analyze_all_gem", analyze_all_gem))

    logger.info("Бот запущено.")
    app.run_polling()
