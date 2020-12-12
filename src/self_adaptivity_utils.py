import numpy as np


def self_adapt_param(p: float,
                     p_min: float,
                     p_max: float,
                     d: float,
                     d_target: float,
                     xi: float
                    ) -> float:
    """ Self adapts a param p given a measure m and control xi

    Args:
        p (float): The param to self adapt
        p_min (float): The lower bound of the param
        p_max (float): The upper bound of the param
        d (float): The measure that affects p
        d_target (float): The desired d value
        xi (float): The parameter controlling the adaptivity strength

    Returns:
        float: The adapted param p
    """
    new_p = max(p_min, min(p_max, p * (1 + xi * (d_target - d) / d)))
    return new_p
