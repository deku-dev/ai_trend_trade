from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes
)
from telegram.constants import ParseMode
from telegram import Update
from app.commands_prompt import prompt_history, set_prompt, show_my_prompt
from app.commands_utils import history
from app.weights_commands import reset_weights, set_weights, show_weights
from config import TELEGRAM_API_KEY
from loguru import logger

from app.commands_gpt import analyze_gpt, analyze_all_gpt
from app.commands_gemini import analyze_gem, analyze_all_gem

logger.add("logs/app.log", format="{time} | {level} | {message}", rotation="10 MB")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привіт! Я бот для аналізу акцій. Ось список доступних команд:\n\n"
        
        "🔍 <b>Основні команди аналізу:</b>\n"
        "/analyze_gem <TICKER> [YYYY-MM-DD] - Аналіз тикера за допомогою Gemini\n"
        "/analyze_gpt <TICKER> [YYYY-MM-DD] - Аналіз тикера за допомогою ChatGPT\n"
        "/analyze_all_gem ticker1,ticker2 [YYYY-MM-DD] - Аналіз кількох тикерів (Gemini)\n"
        "/analyze_all_gpt ticker1,ticker2 [YYYY-MM-DD] - Аналіз кількох тикерів (ChatGPT)\n\n"
        
        "⚙️ <b>Керування промптами (інструкціями для ШІ):</b>\n"
        "/setmyprompt [текст] - Встановити власну інструкцію для аналізу\n"
        "/myprompt - Переглянути поточну інструкцію\n"
        "/prompthistory - Показати історію змін інструкцій\n"
        "/resetmyprompt - Скинути інструкцію до стандартної\n\n"
        
        "⚖️ <b>Керування вагами параметрів:</b>\n"
        "/setmyweights параметр1:значення параметр2:значення - Встановити власні ваги\n"
        "/myweights - Переглянути поточні ваги параметрів\n"
        "/weightshistory - Показати історію змін ваг\n"
        "/resetmyweights - Скинути ваги до стандартних\n\n"
        
        "ℹ️ <b>Додаткові команди:</b>\n"
        "/history - Показати історію аналізів\n"
        "/help - Детальна довідка по командам\n"
        "/status - Перевірити статус системи\n\n"
        
        "<i>Примітка: Параметри в квадратних дужках [] є необов'язковими</i>",
        parse_mode=ParseMode.HTML
    )

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_API_KEY).build()

    app.add_handler(CommandHandler("start", help_command))
    app.add_handler(CommandHandler("analyze_gpt", analyze_gpt))
    app.add_handler(CommandHandler("analyze_gem", analyze_gem))
    app.add_handler(CommandHandler("analyze_all_gpt", analyze_all_gpt))
    app.add_handler(CommandHandler("analyze_all_gem", analyze_all_gem))
    
    app.add_handler(CommandHandler("prompthistory", prompt_history))
    app.add_handler(CommandHandler("myprompt", show_my_prompt))
    app.add_handler(CommandHandler("set_prompt", set_prompt))
    
    app.add_handler(CommandHandler("setweights", set_weights))
    app.add_handler(CommandHandler("myweights", show_weights))
    app.add_handler(CommandHandler("resetweights", reset_weights))
    
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("help", help_command))

    logger.info("Бот запущено.")
    app.run_polling()
