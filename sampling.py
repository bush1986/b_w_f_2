from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gumbel_r, uniform
from scipy.stats._distn_infrastructure import rv_frozen

from config import Settings


def define_random_variables(*, V_b_100: float = 28.5, beta_coeff: float = 0.12) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, rv_frozen]]:
    """Return mean, std and distributions for U10 and alpha."""
    V_design = V_b_100
    beta = beta_coeff * V_design
    mu_loc = V_design + beta * math.log(-math.log(1 - 1 / 100))
    mean_u10 = mu_loc + 0.5772156649 * beta
    std_u10 = math.pi * beta / math.sqrt(6.0)
    mu_alpha = 1.5
    std_alpha = 3.0 / math.sqrt(12.0)
    mu = {"U10": mean_u10, "alpha": mu_alpha}
    std = {"U10": std_u10, "alpha": std_alpha}
    dists = {"U10": gumbel_r(loc=mu_loc, scale=beta), "alpha": uniform(loc=0.0, scale=3.0)}
    return mu, std, dists


def generate_ccd_samples(center: Dict[str, float], std: Dict[str, float], *, k: float = 1.0, alpha_star: float = math.sqrt(2.0), n_center: int = 3) -> pd.DataFrame:
    """Generate a central composite design."""
    delta = {"U10": k * std["U10"], "alpha": k * std["alpha"]}
    recs: List[Dict[str, float]] = []
    for _ in range(n_center):
        recs.append({"U10": center["U10"], "alpha": center["alpha"], "x1": 0.0, "x2": 0.0, "type": "center"})
    for x1 in (-1.0, 1.0):
        for x2 in (-1.0, 1.0):
            recs.append({"U10": center["U10"] + x1 * delta["U10"], "alpha": center["alpha"] + x2 * delta["alpha"], "x1": x1, "x2": x2, "type": "corner"})
    axes = [(alpha_star, 0.0), (-alpha_star, 0.0), (0.0, alpha_star), (0.0, -alpha_star)]
    for x1, x2 in axes:
        recs.append({"U10": center["U10"] + x1 * delta["U10"], "alpha": center["alpha"] + x2 * delta["alpha"], "x1": x1, "x2": x2, "type": "axial"})
    df = pd.DataFrame(recs)
    df.attrs["center"] = center
    df.attrs["delta"] = delta
    return df
