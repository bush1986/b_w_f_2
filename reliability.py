from __future__ import annotations

import ast
import logging
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from sampling import generate_ccd_samples
from surrogate import MultiRSM, QuadraticRSM
from fsi import FSISimulationManager, run_coupled_simulation, run_simulations

logger = logging.getLogger(__name__)


def create_limit_state(expr: str) -> Callable[[float, Dict[str, float]], float]:
    """Safely create a limit-state function from an expression."""
    tree = ast.parse(expr, mode="eval")
    tokens = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def _eval(node: ast.AST, env: Dict[str, float]) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body, env)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return float(env[node.id])
        if isinstance(node, ast.BinOp):
            left = _eval(node.left, env)
            right = _eval(node.right, env)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand, env)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
        raise ValueError("Unsupported expression")

    def func(ucr: float, sample: Dict[str, float]) -> float:
        allowed = set(sample.keys()) | {"Ucr"}
        if not tokens <= allowed:
            raise ValueError(f"Invalid tokens: {tokens - allowed}")
        env = {"Ucr": ucr, **sample}
        return _eval(tree, env)

    return func


def create_normalized_limit_state() -> Callable[[float, Dict[str, float]], float]:
    return lambda ucr, sample: (ucr - sample["U10"]) / ucr


class ReliabilitySolver:
    def __init__(self, g_func: Callable[[float, Dict[str, float]], float]) -> None:
        self.g_func = g_func

    def solve(
        self,
        rsm: QuadraticRSM,
        dists: Dict[str, rv_frozen],
        *,
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> Tuple[float, Dict[str, float], float]:
        mu = np.array([dist.mean() for dist in dists.values()])
        sigma = np.array([dist.std() for dist in dists.values()])
        var_names = list(dists.keys())
        u = np.zeros_like(mu)
        beta = 0.0
        for _ in range(max_iter):
            x = mu + sigma * u
            sample_df = pd.DataFrame([{v: val for v, val in zip(var_names, x)}])
            ucr_pred = rsm.predict(sample_df)[0]
            g = self.g_func(ucr_pred, {v: val for v, val in zip(var_names, x)})
            grad_x = rsm.gradient(sample_df)
            grad_u = grad_x * sigma
            norm_grad = float(np.linalg.norm(grad_u))
            if norm_grad < 1e-12:
                break
            alpha_vec = grad_u / norm_grad
            beta = np.dot(u, alpha_vec) - g / norm_grad
            u_new = beta * alpha_vec
            if np.linalg.norm(u_new - u) < tol and abs(g) < tol:
                u = u_new
                break
            u = u_new
        design_point = {v: mu_i + sigma_i * ui for v, mu_i, sigma_i, ui in zip(var_names, mu, sigma, u)}
        g_pred_design = self.g_func(rsm.predict(pd.DataFrame([design_point]))[0], design_point)
        beta = float(np.linalg.norm(u))
        return beta, design_point, g_pred_design


def update_sampling_center(
    center: Dict[str, float],
    g_true_center: float,
    design_point: Dict[str, float],
    g_pred_design: float,
) -> Dict[str, float]:
    if abs(g_true_center - g_pred_design) < 1e-6:
        return center
    ratio = g_true_center / (g_true_center - g_pred_design)
    return {k: center[k] + ratio * (design_point[k] - center[k]) for k in center}


def iterate_until_convergence(
    fsi_manager: Optional[FSISimulationManager],
    mu: Dict[str, float],
    std: Dict[str, float],
    dists: Dict[str, rv_frozen],
    *,
    g_func: Callable[[float, Dict[str, float]], float],
    max_iter: int = 15,
    tol: float = 0.01,
    beta_tol: float = 0.005,
    pop_size: int = 80,
    n_gen: int = 60,
    use_fsi: bool = True,
    seed: int = 42,
    sampling_cfg: Optional[Dict[str, float]] = None,
) -> Tuple[MultiRSM, List[Dict[str, float]]]:
    center = mu.copy()
    scale = (sampling_cfg or {}).get("initial_scale", 1.0)
    shrink = (sampling_cfg or {}).get("shrink", 0.8)
    min_scale = (sampling_cfg or {}).get("min_scale", 0.2)
    history: List[Dict[str, float]] = []
    prev_beta: Optional[float] = None
    multi_rsm = MultiRSM(pop_size=pop_size, n_gen=n_gen)
    for i in range(max_iter):
        logger.info("迭代 %d/%d - 中心点: U10=%.2f alpha=%.2f scale=%.2f", i + 1, max_iter, center["U10"], center["alpha"], scale)
        samples = generate_ccd_samples(center, std, k=scale)
        sims = run_simulations(fsi_manager, samples.to_dict("records"), base_seed=seed + i * len(samples), use_fsi=use_fsi)
        df_resp = pd.DataFrame(sims)
        if df_resp.isna().any().any():
            logger.warning("仿真结果包含NaN，已用前后值填充")
            df_resp = df_resp.fillna(method="ffill").fillna(method="bfill")
        multi_rsm.fit_all(samples, df_resp)
        solver = ReliabilitySolver(g_func)
        beta, design, g_pred = solver.solve(multi_rsm.models["Ucr"], dists)
        logger.info("可靠度: β=%.4f 设计点: U10=%.2f alpha=%.2f", beta, design["U10"], design["alpha"])
        if (prev_beta is not None and abs(beta - prev_beta) < beta_tol) or (
            history and np.linalg.norm(np.array(list(design.values())) - np.array(list(history[-1].values()))) < tol
        ):
            history.append(design)
            logger.info("达到收敛条件")
            break
        prev_beta = beta
        history.append(design)
        center_res = run_coupled_simulation(fsi_manager, center, use_fsi=use_fsi, seed=seed)
        g_true_center = g_func(center_res["Ucr"], center)
        logger.info("中心点验证 g_true=%.4f g_pred=%.4f", g_true_center, g_pred)
        center = update_sampling_center(center, g_true_center, design, g_pred)
        if abs(g_true_center) < 1e-3:
            logger.info("中心点 g 值接近 0, 停止迭代")
            break
        scale = max(scale * shrink, min_scale)
    return multi_rsm, history
