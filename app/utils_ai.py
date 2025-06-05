import json, os

import pandas as pd
from datetime import datetime

FEATURES_PATH = 'data/features.csv'
HISTORY_PATH = 'output/history.json'
DATA_PATH = 'data/tickers.csv'

OUTPUT_DIR = 'output/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Ініціалізація файлу історії
if not os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'w') as f:
        json.dump({}, f)


def load_history():
    with open(HISTORY_PATH, 'r') as f:
        return json.load(f)

# Збереження історії
def save_history(history):
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)

# Перевірка, чи вже є запис для конкретної дати
def is_already_processed(ticker):
    history = load_history()
    today = datetime.now().strftime('%Y-%m-%d')
    if ticker in history and today in history[ticker]:
        return True
    return False

# Додавання нового запису в історію (з результатом за замовчуванням None)
def add_to_history(ticker, date_str=None):
    history = load_history()
    
    if ticker not in history:
        history[ticker] = {}
    
    if date_str not in history[ticker]:
        history[ticker][date_str] = {"result": None}
    
    save_history(history)

# Оновлення результату для запису
def update_history(ticker, result, date_str=None):
    history = load_history()
    
    if ticker in history and date_str in history[ticker]:
        history[ticker][date_str]["result"] = result
        save_history(history)

def load_features():
    df = pd.read_csv(FEATURES_PATH)
    features = dict(zip(df['parameter'], df['weight']))
    return features

# Load tickers from CSV file
def load_tickers():
    return pd.read_csv(DATA_PATH)['ticker'].tolist()

def validate_date(date_str: str) -> datetime | None:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None
    
def update_feedback(ticker: str, date: str, feedback: str) -> None:
    history = load_history()
    if ticker in history and date in history[ticker]:
        history[ticker][date]["feedback"] = feedback
        save_history(history)