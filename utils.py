import pandas as pd
import numpy as np

def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Додає в DataFrame стовпці:
      - DI+   (Positive Directional Indicator)
      - DI-   (Negative Directional Indicator)
      - ADX   (Average Directional Index)
    
    Параметри:
    -----------
    df : pd.DataFrame
        Повинен містити колонки ['h','l','c'] — high, low, close.
    window : int
        Період для обчислення (за замовчуванням 14).
    
    Повертає:
    ---------
    result : pd.DataFrame
        Копія вхідного df з доданими стовпцями ['DI+', 'DI-', 'ADX'].
    """
    data = df.copy()

    # Крок 1: Directional Moves
    up_move   = data['h'].diff()
    down_move = -data['l'].diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0),  up_move,   0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Крок 2: True Range
    high_low    = data['h'] - data['l']
    high_close  = (data['h'] - data['c'].shift()).abs()
    low_close   = (data['l'] - data['c'].shift()).abs()
    true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Крок 3: Wilder’s smoothing (EWMA з α=1/window)
    atr = true_range.ewm(alpha=1/window, adjust=False).mean()
    plus_dm_smooth  = pd.Series(plus_dm,  index=data.index).ewm(alpha=1/window, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=data.index).ewm(alpha=1/window, adjust=False).mean()

    # Крок 4: DI+ і DI-
    data['DI+'] = 100 * (plus_dm_smooth  / atr)
    data['DI-'] = 100 * (minus_dm_smooth / atr)

    # Крок 5: DX і ADX
    dx  = (data['DI+'] - data['DI-']).abs() / (data['DI+'] + data['DI-']) * 100
    data['ADX'] = dx.ewm(alpha=1/window, adjust=False).mean()

    return data