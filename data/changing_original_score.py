import re
import numpy as np
import pandas as pd
def parse_original_score(score):
    if pd.isna(score):
        return np.nan

    score = str(score).strip().lower()

    # Przypadek: X/Y
    match = re.match(r'^([\d\.]+)\s*/\s*([\d\.]+)$', score)
    if match:
        try:
            num = float(match.group(1).rstrip('.'))
            denom = float(match.group(2).rstrip('.'))
            if denom == 0:
                return np.nan
            return (num / denom) * 10
        except ValueError:
            return np.nan

    # Przypadek: "X out of Y"
    match = re.match(r'^([\d\.]+)\s*out of\s*([\d\.]+)$', score)
    if match:
        try:
            num = float(match.group(1).rstrip('.'))
            denom = float(match.group(2).rstrip('.'))
            if denom == 0:
                return np.nan
            return (num / denom) * 10
        except ValueError:
            return np.nan

    # Przypadek: "X stars"
    match = re.match(r'^([\d\.]+)\s*stars?$', score)
    if match:
        try:
            return float(match.group(1).rstrip('.')) * 2  # np. 5 stars = 10
        except ValueError:
            return np.nan

    return np.nan
