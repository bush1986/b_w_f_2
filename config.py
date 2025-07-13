from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

try:
    from pydantic import BaseSettings
except Exception:  # pragma: no cover - missing dependency
    BaseSettings = object  # type: ignore[misc]


class Settings(BaseSettings):
    wbpj_template: str = "bridge_fsi_template.wbpj"
    working_dir: str = "fsi_simulations"
    random_vars: Dict[str, float] = {"V_b_100": 28.5, "beta_coeff": 0.12}
    nsga_pop: int = 80
    nsga_gen: int = 60
    sampling: Dict[str, float] = {"initial_scale": 1.0, "shrink": 0.8, "min_scale": 0.2}
    random_seed: int = 42
    limit_state: str = "Ucr - U10"
    convergence: Dict[str, float] = {"max_iter": 15, "tol": 0.01, "beta_tol": 0.005}
    thresholds: Dict[str, Tuple[str, float]] = {}
    thresholds: Dict[str, Tuple[str, float]] = {
        "DS1": ("Ucr_norm", 0.25),
        "DS2": ("Ucr_norm", 0.50),
        "DS3": ("Ucr_norm", 0.75),
        "DS4": ("Ucr_norm", 1.00),
        "Displacement_DS1": ("sigma_q", 0.1),
        "Displacement_DS2": ("sigma_q", 0.2),
        "Displacement_DS3": ("sigma_q", 0.3),
        "Acceleration_DS1": ("sigma_a", 0.05),
        "Acceleration_DS2": ("sigma_a", 0.1),
        "Acceleration_DS3": ("sigma_a", 0.15),
    }
    use_fsi: bool = True
    flutter: Dict[str, Any] = {"case_dir": "flutter_cases", "wind_speeds": [20, 25, 30, 35, 40, 45, 50, 55, 60]}
    bridge_params: Dict[str, float] = {
        "mass": 1500,
        "inertia": 2.35,
        "f_h": 0.10491093,
        "f_alpha": 0.12921779,
        "damping_h": 0.005,
        "damping_alpha": 0.005,
        "width": 10,
        "density": 1.25,
    }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
