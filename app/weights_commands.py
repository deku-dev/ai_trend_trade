from telegram import Update
from telegram.ext import ContextTypes, CommandHandler
from app.weights_manager import save_weights, get_weights_by_user_id, reset_user_weights, get_default_weights

async def set_weights(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command: /setweights trend:0.5 momentum:0.3 volume:0.15 fundamentals:0.05"""
    user_id = update.effective_user.id
    args = context.args
    
    if not args:
        await update.message.reply_text("Please provide weights in format: feature:value")
        return
    
    weights = {}
    for arg in args:
        if ':' in arg:
            feature, value = arg.split(':', 1)
            try:
                weights[feature.strip()] = float(value.strip())
            except ValueError:
                await update.message.reply_text(f"Invalid value for {feature}: {value}")
                return
    
    save_weights(user_id, weights)
    await update.message.reply_text("Your custom weights have been saved!")

async def show_weights(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command: /myweights"""
    user_id = update.effective_user.id
    weights = get_weights_by_user_id(user_id) or get_default_weights()
    
    response = "Your current weights:\n"
    for feature, value in weights.items():
        response += f"- {feature}: {value}\n"
    
    await update.message.reply_text(response)

async def reset_weights(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command: /resetweights"""
    user_id = update.effective_user.id
    if reset_user_weights(user_id):
        await update.message.reply_text("Your weights have been reset to default")
    else:
        await update.message.reply_text("No custom weights to reset")