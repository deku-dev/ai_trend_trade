# weights_manager.py
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

# Constants
WEIGHTS_FILE = "data/weights.json"
DEFAULT_WEIGHTS_PATH = 'data/features.csv'

# Ensure data directory exists
Path("data").mkdir(exist_ok=True)

def _load_default_weights() -> Dict[str, float]:
    """Load default weights from CSV file or return empty dict if not found."""
    if not os.path.exists(DEFAULT_WEIGHTS_PATH):
        return {}
    
    try:
        df = pd.read_csv(DEFAULT_WEIGHTS_PATH)
        return dict(zip(df['parameter'], df['weight']))
    except Exception:
        return {}

def _load_weights() -> Dict:
    """Load weights from JSON file or return initial structure if file doesn't exist."""
    default_weights = _load_default_weights()
    
    if not os.path.exists(WEIGHTS_FILE):
        return {
            "default": default_weights,
            "users": {}
        }
    
    try:
        with open(WEIGHTS_FILE, 'r') as f:
            data = json.load(f)
            # Ensure backward compatibility
            if "users" not in data:
                data["users"] = {}
            if "default" not in data:
                data["default"] = default_weights
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        return {
            "default": default_weights,
            "users": {}
        }

def _save_weights(weights: Dict) -> None:
    """Save weights to JSON file."""
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(weights, f, indent=4)

def get_default_weights() -> Dict[str, float]:
    """Return the default weights."""
    weights = _load_weights()
    return weights.get("default", {})

def get_weights_by_user_id(user_id: Union[str, int]) -> Optional[Dict[str, float]]:
    """
    Get the latest custom weights for a specific user.
    Returns None if no custom weights exist for the user.
    """
    user_id = str(user_id)
    weights = _load_weights()
    user_data = weights["users"].get(user_id)
    
    if not user_data or not user_data["history"]:
        return None
    
    # Return the most recent weights (last in the history list)
    return user_data["history"][-1]["weights"]

def get_weights_history(user_id: Union[str, int]) -> List[Dict]:
    """
    Get the full weights history for a user including timestamps.
    Returns empty list if no history exists.
    """
    user_id = str(user_id)
    weights = _load_weights()
    user_data = weights["users"].get(user_id)
    return user_data["history"] if user_data else []

def save_weights(user_id: Union[str, int], weights: Dict[str, float]) -> None:
    """
    Save new weights for a specific user with timestamp.
    Maintains history of all previous weight configurations.
    """
    user_id = str(user_id)
    weights_data = _load_weights()
    
    if "users" not in weights_data:
        weights_data["users"] = {}
    
    if user_id not in weights_data["users"]:
        weights_data["users"][user_id] = {"history": []}
    
    # Add new weights entry to history
    new_entry = {
        "weights": weights,
        "timestamp": datetime.now().isoformat()
    }
    
    weights_data["users"][user_id]["history"].append(new_entry)
    _save_weights(weights_data)

def get_active_weights(user_id: Optional[Union[str, int]] = None) -> Dict[str, float]:
    """
    Get the active weights for a user (latest custom if exists, otherwise default).
    If no user_id is provided, returns the default weights.
    """
    if user_id:
        custom_weights = get_weights_by_user_id(user_id)
        if custom_weights:
            return custom_weights
    return get_default_weights()

def reset_user_weights(user_id: Union[str, int]) -> bool:
    """
    Reset a user's weights history (delete all custom weights).
    Returns True if reset was successful, False if user had no history.
    """
    user_id = str(user_id)
    weights_data = _load_weights()
    
    if user_id in weights_data["users"]:
        del weights_data["users"][user_id]
        _save_weights(weights_data)
        return True
    return False

def format_weights_for_prompt(weights: Dict[str, float]) -> str:
    """
    Format weights for use in AI prompts.
    Returns a string like:
        - Feature1: weight X
        - Feature2: weight Y
    """
    return "\n".join(f"- {feature}: weight {weight}" for feature, weight in weights.items())