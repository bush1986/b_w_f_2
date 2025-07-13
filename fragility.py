from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import kstest, lognorm
from scipy.stats._distn_infrastructure import rv_frozen

from surrogate import MultiRSM

logger = logging.getLogger(__name__)


def compute_capacity(
    rsm: MultiRSM,
    dists: Dict[str, rv_frozen],
    indicator: str,
    thresh: float,
    *,
    size: int = 1000,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("计算容量 %s", indicator)
    rng = np.random.default_rng(seed)
    alpha_samples = dists["alpha"].rvs(size=size, random_state=rng)
    mu_u = dists["U10"].mean()
    std_u = dists["U10"].std()
    capacities: List[float] = []
    for a in alpha_samples:
        if indicator in ("Ucr", "Ucr_norm"):
            def func(u: float) -> float:
                df = pd.DataFrame({"U10": [u], "alpha": [a]})
                ucr_pred = rsm.predict("Ucr", df)[0]
                if indicator == "Ucr_norm":
                    return (ucr_pred - u) / ucr_pred - thresh
                return ucr_pred - u
            u_low = 1.0
            u_high = mu_u + 6 * std_u
            f_low = func(u_low)
            f_high = func(u_high)
            if f_low * f_high > 0:
                cap = math.nan
            else:
                try:
                    cap = brentq(func, u_low, u_high, xtol=0.01, maxiter=100)
                except Exception:
                    cap = math.nan
            capacities.append(cap)
            continue
        grid = np.linspace(1.0, mu_u + 6 * std_u, 120)
        df = pd.DataFrame({"U10": grid, "alpha": np.full_like(grid, a)})
        val = rsm.predict(indicator, df)
        g = val - thresh
        if g[0] >= 0:
            capacities.append(1.0)
            continue
        if g[-1] <= 0:
            capacities.append(math.nan)
            continue
        idx = int(np.where(g >= 0)[0][0])
        cap = np.interp(0.0, [g[idx - 1], g[idx]], [grid[idx - 1], grid[idx]])
        capacities.append(cap)
    return np.array(capacities), alpha_samples


def build_all_capacities(
    dists: Dict[str, rv_frozen],
    rsm: MultiRSM,
    thresholds: Dict[str, Tuple[str, float]],
    *,
    size: int = 1000,
    seed: int | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    logger.info("构建容量样本")
    return {label: compute_capacity(rsm, dists, ind, thr, size=size, seed=seed) for label, (ind, thr) in thresholds.items()}


def fit_fragility_curve(capacity_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Tuple[float, float, float, float]]:
    logger.info("拟合脆弱性曲线")
    results: Dict[str, Tuple[float, float, float, float]] = {}
    for label, (samples, _) in capacity_dict.items():
        samples = samples[~np.isnan(samples)]
        if len(samples) == 0:
            results[label] = (math.nan, math.nan, math.nan, math.nan)
            logger.warning("指标 %s 无有效样本", label)
            continue
        shape, loc, scale = lognorm.fit(samples, floc=0)
        theta = scale
        beta = shape
        ks_stat, _ = kstest(np.log(samples), "norm", args=(math.log(theta), beta))
        se_beta = beta / math.sqrt(2 * len(samples)) if len(samples) > 1 else 0.0
        results[label] = (theta, beta, se_beta, ks_stat)
        logger.info("%s: θ=%.4f β=%.4f", label, theta, beta)
    return results


def compute_annual_pf(
    dists: Dict[str, rv_frozen],
    cap_pair: Tuple[np.ndarray, np.ndarray],
    *,
    n_mc: int = 10000,
    seed: int = 42,
) -> float:
    caps, alphas = cap_pair
    valid = ~np.isnan(caps)
    if not np.any(valid):
        return math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, valid.sum(), size=n_mc)
    cap_samples = caps[valid][idx]
    _ = alphas[valid][idx]
    u10_samples = dists["U10"].rvs(size=n_mc, random_state=rng)
    return float(np.mean(u10_samples > cap_samples))
