# commands_prompt.py
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler
from app.prompt_manager import save_prompt, get_prompt_history, get_active_prompt
from datetime import datetime

async def set_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command: /setmyprompt <prompt text>"""
    user_id = update.effective_user.id
    prompt_text = " ".join(context.args)
    
    if not prompt_text:
        await update.message.reply_text("Please provide a prompt text")
        return
    
    save_prompt(user_id, prompt_text)
    await update.message.reply_text("Your custom prompt has been saved!")

async def show_my_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command: /myprompt"""
    user_id = update.effective_user.id
    prompt = get_active_prompt(user_id)
    
    await update.message.reply_text(
        f"Your current prompt:\n\n{prompt}",
        parse_mode="Markdown"
    )

async def prompt_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command: /prompthistory"""
    user_id = update.effective_user.id
    history = get_prompt_history(user_id)
    
    if not history:
        await update.message.reply_text("You have no prompt history")
        return
    
    response = "Your prompt history:\n\n"
    for i, entry in enumerate(history, 1):
        date = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M")
        response += f"{i}. [{date}]:\n{entry['prompt'][:100]}...\n\n"
    
    await update.message.reply_text(response)