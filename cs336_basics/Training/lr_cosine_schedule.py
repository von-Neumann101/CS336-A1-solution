import math


def lr_cosine_schedule(
        t: int,
        lr_max: float,
        lr_min: float,
        T_w: int,
        T_c: int
) -> float:
    if t < T_w:
        return lr_max * t / T_w
    elif t >= T_w and t <= T_c:
        return lr_min + 0.5 * (1 + math.cos((t - T_w) * math.pi / (T_c - T_w))) * (lr_max - lr_min)
    else:
        return lr_min