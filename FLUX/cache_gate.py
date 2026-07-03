from typing import Optional, Tuple


def decide_cache_refresh(
    *,
    gate: str,
    distance: float,
    acc_before: float,
    delta_acc: float,
    delta_single: Optional[float] = None,
) -> Tuple[bool, float, str, bool, bool]:
    acc_with_distance = acc_before + distance
    acc_hit = not (acc_with_distance < delta_acc)
    single_hit = gate == "dual" and delta_single is not None and distance > delta_single

    if single_hit:
        return True, 0.0, "single", True, acc_hit
    if acc_hit:
        return True, 0.0, "acc", False, True
    return False, acc_with_distance, "skip", False, False
