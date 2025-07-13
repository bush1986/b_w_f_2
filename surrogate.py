from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

import logging

logger = logging.getLogger(__name__)


def _log_exc(msg: str, exc: Exception) -> None:
    logger.error("%s: %s", msg, exc, exc_info=True)


@dataclass
class QuadraticRSM:
    coeff: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.poly = PolynomialFeatures(degree=2, include_bias=True)
        self.center: Dict[str, float] | None = None
        self.delta: Dict[str, float] | None = None
        self.var_names: List[str] | None = None
        self.model = Ridge(alpha=0.0, fit_intercept=False)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(
        self,
        samples: pd.DataFrame,
        responses: np.ndarray,
        *,
        center: Dict[str, float],
        delta: Dict[str, float],
        alpha: float = 0.0,
    ) -> "QuadraticRSM":
        self.center = center
        self.delta = delta
        self.var_names = list(center.keys())
        self.model.alpha = alpha
        self.X = self.poly.fit_transform(samples[[f"x{i+1}" for i in range(len(samples.columns) - 3)]])
        self.y = responses
        self.model.fit(self.X, self.y)
        self.coeff = self.model.coef_.copy()
        return self

    def optimize(self, *, pop_size: int = 80, n_gen: int = 60) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("模型未初始化")
        n_coef = self.X.shape[1]

        class RSMProblem(ElementwiseProblem):
            def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
                super().__init__(n_var=n_coef, n_obj=2, xl=-10.0, xu=10.0)
                self.X = X
                self.y = y

            def _evaluate(self, x: np.ndarray, out: Dict[str, np.ndarray]) -> None:
                pred = self.X @ x
                mse = float(((pred - self.y) ** 2).mean())
                l2 = float(np.linalg.norm(x))
                out["F"] = np.array([mse, l2])

        ref_dirs = get_reference_directions("das-dennis", 2, n_points=pop_size)
        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
        problem = RSMProblem(self.X, self.y)
        res = minimize(problem, algo, ("n_gen", n_gen), verbose=False)
        F = res.F
        F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-12)
        g_val = (F_norm[:, 0] + F_norm[:, 1]) / math.sqrt(2.0)
        curvature = np.sqrt((F_norm ** 2).sum(axis=1)) - g_val
        best = res.X[int(np.argmax(curvature))]
        self.coeff = best
        self.model.coef_ = best

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.coeff is None:
            raise RuntimeError("模型未初始化")
        X = self.poly.transform(df)
        return X @ self.coeff

    def gradient(self, df: pd.DataFrame) -> np.ndarray:
        if self.coeff is None:
            raise RuntimeError("模型未初始化")
        X = self.poly.transform(df)
        grad = self.poly.powers_[:, 1:] * self.coeff[:, None]
        return (X[:, :, None] * grad).sum(axis=1)


class MultiRSM:
    def __init__(self, pop_size: int = 80, n_gen: int = 60) -> None:
        self.models = {k: QuadraticRSM() for k in ("Ucr", "sigma_q", "sigma_a")}
        self.pop_size = pop_size
        self.n_gen = n_gen

    def fit_all(self, samples_df: pd.DataFrame, targets_df: pd.DataFrame) -> "MultiRSM":
        center = samples_df.attrs["center"]
        delta = samples_df.attrs["delta"]
        for col in targets_df.columns:
            m = self.models[col]
            m.fit(samples_df, targets_df[col].values, center=center, delta=delta)
            if self.pop_size > 0 and self.n_gen > 0:
                try:
                    m.optimize(pop_size=self.pop_size, n_gen=self.n_gen)
                except Exception as exc:  # pragma: no cover - optimization failure
                    _log_exc("NSGA-III 优化失败", exc)
        return self

    def predict(self, indicator: str, df: pd.DataFrame) -> np.ndarray:
        return self.models[indicator].predict(df)
