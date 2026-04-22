# Bounds for normalizing targets to [-1, 1] (start_time_sec, weight_db, duration_sec)
BOUNDS_START_TIME = (0.0, 20.0)
BOUNDS_WEIGHT_DB = (-40.0, 0.0)
BOUNDS_DURATION = (0.0, 15.0)


def _normalize_to_minus1_1(x: float, low: float, high: float) -> float:
    """Map value from [low, high] to [-1, 1]. Clamp if outside range."""
    if high <= low:
        return 0.0
    x = max(low, min(high, float(x)))
    return 2.0 * (x - low) / (high - low) - 1.0


def _denormalize_from_minus1_1(x_norm: float, low: float, high: float) -> float:
    """Map value from [-1, 1] back to [low, high]."""
    if high <= low:
        return low
    t = (float(x_norm) + 1.0) / 2.0
    return low + t * (high - low)


def normalize_targets(start_time: float, weight_db: float, duration: float) -> tuple:
    """Return (start_norm, weight_norm, duration_norm) in [-1, 1]."""
    return (
        _normalize_to_minus1_1(start_time, BOUNDS_START_TIME[0], BOUNDS_START_TIME[1]),
        _normalize_to_minus1_1(weight_db, BOUNDS_WEIGHT_DB[0], BOUNDS_WEIGHT_DB[1]),
        _normalize_to_minus1_1(duration, BOUNDS_DURATION[0], BOUNDS_DURATION[1]),
    )


def denormalize_outputs(
    start_norm: float, weight_norm: float, duration_norm: float
) -> tuple:
    """Map model outputs from [-1, 1] back to original scale."""
    return (
        _denormalize_from_minus1_1(start_norm, BOUNDS_START_TIME[0], BOUNDS_START_TIME[1]),
        _denormalize_from_minus1_1(weight_norm, BOUNDS_WEIGHT_DB[0], BOUNDS_WEIGHT_DB[1]),
        _denormalize_from_minus1_1(duration_norm, BOUNDS_DURATION[0], BOUNDS_DURATION[1]),
    )
