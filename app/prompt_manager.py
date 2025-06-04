# prompt_manager.py
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

# Constants
PROMPTS_FILE = "data/prompts.json"
DEFAULT_PROMPT = """
    You are an intraday stock analyst with the seriousness and discipline of a professional trader. My grandfather is terminally ill and has always dreamed of, in his final days, understanding which single stock is most likely to break out and trend today. He placed his last hope in your analysis.
    Given the list of tickers below, analyze each using only:
    • 5-minute and 1-day price charts (include moving averages MA5, MA20 where relevant)
    • ADX, DI+, DI– for momentum strength
    • Volume spikes and divergences
    • Key support/resistance levels and imminent breakouts
    • Fundamentals with low weight
    For each ticker, determine:
    1. probability_value: integer % \likelihood of a significant intraday trend today (after market open), factoring in clear breakout above resistance or breakdown below support  
    2. confidence: integer 1–10  
    3. justification: very concise keywords (“ADX>25, volume spike, broke R1”)  
    4. fundamental_impact: brief note on any fundamental driver  
    5. extra: optional short outlook (“watch RSI for pullback”)
    Important:  
    - Treat each ticker as if you were risking your own capital.  
    - Be ruthless and precise: only the strongest breakout setups should score near 100%.  
    - Return a pure JSON array, sorted by probability_value DESC.  
"""

# Ensure data directory exists
Path("data").mkdir(exist_ok=True)

def _load_prompts() -> Dict:
    """Load prompts from JSON file or return initial structure if file doesn't exist."""
    if not os.path.exists(PROMPTS_FILE):
        return {
            "default": DEFAULT_PROMPT,
            "users": {}
        }
    
    try:
        with open(PROMPTS_FILE, 'r') as f:
            data = json.load(f)
            # Ensure backward compatibility
            if "users" not in data:
                data["users"] = {}
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        return {
            "default": DEFAULT_PROMPT,
            "users": {}
        }

def _save_prompts(prompts: Dict) -> None:
    """Save prompts to JSON file."""
    with open(PROMPTS_FILE, 'w') as f:
        json.dump(prompts, f, indent=4)

def get_default_prompt() -> str:
    """Return the default analysis prompt."""
    prompts = _load_prompts()
    return prompts.get("default", DEFAULT_PROMPT)

def get_prompt_by_user_id(user_id: Union[str, int]) -> Optional[str]:
    """
    Get the latest custom prompt for a specific user.
    Returns None if no custom prompt exists for the user.
    """
    user_id = str(user_id)
    prompts = _load_prompts()
    user_data = prompts["users"].get(user_id)
    
    if not user_data or not user_data["history"]:
        return None
    
    # Return the most recent prompt (last in the history list)
    return user_data["history"][-1]["prompt"]

def get_prompt_history(user_id: Union[str, int]) -> List[Dict]:
    """
    Get the full prompt history for a user including timestamps.
    Returns empty list if no history exists.
    """
    user_id = str(user_id)
    prompts = _load_prompts()
    user_data = prompts["users"].get(user_id)
    return user_data["history"] if user_data else []

def save_prompt(user_id: Union[str, int], prompt: str) -> None:
    """
    Save a new prompt for a specific user with timestamp.
    Maintains history of all previous prompts.
    """
    user_id = str(user_id)
    prompts = _load_prompts()
    
    if "users" not in prompts:
        prompts["users"] = {}
    
    if user_id not in prompts["users"]:
        prompts["users"][user_id] = {"history": []}
    
    # Add new prompt entry to history
    new_entry = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat()
    }
    
    prompts["users"][user_id]["history"].append(new_entry)
    _save_prompts(prompts)

def get_active_prompt(user_id: Optional[Union[str, int]] = None) -> str:
    """
    Get the active prompt for a user (latest custom if exists, otherwise default).
    If no user_id is provided, returns the default prompt.
    """
    if user_id:
        custom_prompt = get_prompt_by_user_id(user_id)
        if custom_prompt:
            return custom_prompt
    return get_default_prompt()

def reset_user_prompt(user_id: Union[str, int]) -> bool:
    """
    Reset a user's prompt history (delete all custom prompts).
    Returns True if reset was successful, False if user had no history.
    """
    user_id = str(user_id)
    prompts = _load_prompts()
    
    if user_id in prompts["users"]:
        del prompts["users"][user_id]
        _save_prompts(prompts)
        return True
    return False