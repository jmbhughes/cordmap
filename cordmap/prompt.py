from typing import List

def get_suvi_prompt_box() -> List[int]:
    """Retrieve the SUVI prompt box

    Returns:
        List[int, int, int, int]: x_left, y_left, x_right, y_right
    """
    return [24, 24, 1000, 1000]