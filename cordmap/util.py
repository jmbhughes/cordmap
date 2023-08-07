from typing import List
from datetime import datetime

def generate_spaced_times(start: datetime, end: datetime, n: int) -> List[datetime]:
    """Generates `n` times between `start` and `end` evenly spaced

    Args:
        start (datetime): beginning time (inclusive)
        end (datetime): ending time (not included)
        count (int): how many times to create between them

    Returns:
        List[datetime]: A list of `n` evenly spaced datetimes between the observations
    """
    step = (end - start) / n
    return [start + i * step for i in range(n)]