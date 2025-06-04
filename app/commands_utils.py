from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ContextTypes
)
from app.utils_ai import load_history

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    history_data = load_history()
    
    if not history_data:
        await update.message.reply_text("Історія аналізів порожня.")
        return
    
    response = "📜 <b>Історія аналізів:</b>\n\n"
    for ticker, dates in history_data.items():
        response += f"<b>{ticker}</b>:\n"
        for date, analysis in dates.items():
            result = analysis.get("result", {})
            prob = result.get("intraday_trend_movement_probability", {}).get("probability_value", "N/A")
            response += f"  - {date}: {prob}\n"
        response += "\n"
    
    await update.message.reply_text(response, parse_mode=ParseMode.HTML)