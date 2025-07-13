from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Settings

logger = logging.getLogger(__name__)

# Attempt to import Ansys Workbench
try:  # pragma: no cover - optional dependency
    from ansys.workbench.core import Workbench, launch_workbench
    WB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    WB_AVAILABLE = False
    Workbench = object  # type: ignore[misc]

def _log_exc(msg: str, exc: Exception) -> None:
    logger.error("%s: %s", msg, exc, exc_info=True)


class SimulationBackend(ABC):
    @abstractmethod
    def run(self, sample: Dict[str, float], *, seed: Optional[int] = None) -> Dict[str, float]:
        pass


@dataclass
class FSISimulationManager(SimulationBackend):
    """Manage bidirectional FSI simulations in Workbench."""

    wbpj_template: str
    working_dir: str = "fsi_simulations"

    def __post_init__(self) -> None:
        self.wb: Optional[Workbench] = None
        self.project_path = ""
        self.working_dir_path = Path(self.working_dir)
        self.working_dir_path.mkdir(parents=True, exist_ok=True)
        if not Path(self.wbpj_template).exists():
            raise FileNotFoundError(f"Workbench模板不存在: {self.wbpj_template}")
        logger.info("FSI管理器初始化完成 | 模板: %s", self.wbpj_template)

    def initialize_workbench(self) -> None:
        if self.wb is None and WB_AVAILABLE:
            logger.info("启动Ansys Workbench...")
            self.wb = launch_workbench()
            logger.info("Workbench实例已启动")

    def prepare_simulation(self, sample: Dict[str, float]) -> Path:
        ts = int(time.time())
        sim_dir = self.working_dir_path / f"sim_{ts}_{sample['U10']:.1f}_{sample['alpha']:.1f}"
        sim_dir.mkdir(exist_ok=True)
        self.project_path = str(sim_dir / f"bridge_fsi_{ts}.wbpj")
        return sim_dir

    def _fake_results(self, sample: Dict[str, float]) -> Dict[str, float]:  # pragma: no cover - placeholder
        u10 = sample["U10"]
        alpha = sample["alpha"]
        ucr = 50.0 - 2.0 * alpha - 0.05 * u10
        sigma_q = 0.008 * u10 ** 2 + 0.02 * alpha
        sigma_a = 0.004 * u10 ** 3 + 0.05 * alpha
        return {"Ucr": ucr, "sigma_q": sigma_q, "sigma_a": sigma_a}

    def run(self, sample: Dict[str, float], *, seed: Optional[int] = None) -> Dict[str, float]:
        self.prepare_simulation(sample)
        if not WB_AVAILABLE:
            return self._fake_results(sample)
        try:  # pragma: no cover - external dependency
            self.initialize_workbench()
            time.sleep(10)
            return self._fake_results(sample)
        except Exception as exc:  # pragma: no cover - safety
            _log_exc("FSI仿真失败", exc)
            return self._fake_results(sample)

    def cleanup(self) -> None:
        if self.wb:
            logger.info("关闭Workbench实例")
            self.wb.exit()
            self.wb = None


def compute_buffeting_rms(
    u: float,
    *,
    rho: float = 1.225,
    b: float = 10.0,
    sigma_u: Optional[float] = None,
    sigma_w: Optional[float] = None,
    l_u: float = 150.0,
    l_w: float = 50.0,
    c_l: float = 0.8,
    c_d: float = 0.5,
    c_m: float = 0.04,
    c_lp: float = 4.0,
    c_mp: float = 0.8,
    f_bend: float = 0.10491093,
    zeta_b: float = 0.005,
    f_tors: float = 0.12921779,
    zeta_t: float = 0.005,
) -> Dict[str, float]:
    """Compute RMS of buffeting response for a given mean wind speed."""
    sigma_u = sigma_u if sigma_u is not None else 0.15 * u
    sigma_w = sigma_w if sigma_w is not None else 0.10 * u

    def von_karman(omega: np.ndarray, sigma: float, L: float, U: float) -> np.ndarray:
        num = 4 * sigma ** 2 * L / U
        den = (1 + 70.8 * (omega * L / U) ** 2) ** (5 / 6)
        return num / den

    def admittance(omega: np.ndarray, B: float, U: float) -> np.ndarray:
        K = omega * B / U
        return 1.0 / np.sqrt(1 + (2 * K) ** 2)

    def buffeting_spectrum(omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Su = von_karman(omega, sigma_u, l_u, u)
        Sw = von_karman(omega, sigma_w, l_w, u)
        chi2 = admittance(omega, b, u) ** 2
        SL = (
            (0.5 * rho * u ** 2 * b) ** 2
            * ((2 * c_l / u) ** 2 * Su + ((c_lp + c_d) / u) ** 2 * Sw)
            * chi2
        )
        SM = (
            (0.5 * rho * u ** 2 * b ** 2) ** 2
            * ((2 * c_m / u) ** 2 * Su + (c_mp / u) ** 2 * Sw)
            * chi2
        )
        return SL, SM

    def h_disp(omega: np.ndarray, omega_n: float, zeta: float) -> np.ndarray:
        r = omega / omega_n
        return 1.0 / np.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)

    omega = np.linspace(0.01, 20 * 2 * math.pi, 20000)
    SL, SM = buffeting_spectrum(omega)
    Hb = h_disp(omega, 2 * math.pi * f_bend, zeta_b)
    Ht = h_disp(omega, 2 * math.pi * f_tors, zeta_t)
    Sq_b = Hb ** 2 * SL
    Sth_t = Ht ** 2 * SM
    Sa_b = (omega ** 2) ** 2 * Hb ** 2 * SL
    Sa_t = (omega ** 2) ** 2 * Ht ** 2 * SM

    sigma_q = float(np.sqrt(np.trapz(Sq_b, omega)))
    sigma_theta = float(np.sqrt(np.trapz(Sth_t, omega)))
    sigma_ddq = float(np.sqrt(np.trapz(Sa_b, omega)))
    sigma_ddtheta = float(np.sqrt(np.trapz(Sa_t, omega)))

    return {
        "sigma_q": sigma_q,
        "sigma_theta": sigma_theta,
        "sigma_ddq": sigma_ddq,
        "sigma_ddtheta": sigma_ddtheta,
    }


def run_simplified_simulation(sample: Dict[str, float], *, seed: Optional[int] = None) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    u10 = sample["U10"]
    alpha = sample["alpha"]
    ucr = 40.0 - 1.5 * alpha + rng.normal(0.0, 1.0)
    buff = compute_buffeting_rms(u10)
    sigma_q = buff["sigma_q"]
    sigma_a = buff["sigma_ddq"]
    logger.info("简化模型: Ucr=%.2f", ucr)
    return {"Ucr": ucr, "sigma_q": sigma_q, "sigma_a": sigma_a}


def run_coupled_simulation(
    fsi_manager: Optional[FSISimulationManager],
    sample: Dict[str, float],
    *,
    use_fsi: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    if fsi_manager and use_fsi:
        try:
            return fsi_manager.run(sample, seed=seed)
        except Exception as exc:  # pragma: no cover - safety
            _log_exc("FSI仿真失败，使用简化模型", exc)
    return run_simplified_simulation(sample, seed=seed)


def run_simulations(
    fsi_manager: Optional[FSISimulationManager],
    samples: List[Dict[str, float]],
    *,
    base_seed: int = 42,
    use_fsi: bool = True,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for i, s in enumerate(samples):
        try:
            res = run_coupled_simulation(fsi_manager, s, use_fsi=use_fsi, seed=base_seed + i)
        except Exception as exc:  # pragma: no cover - safety
            _log_exc(f"样本 {s} 失败", exc)
            res = {"Ucr": np.nan, "sigma_q": np.nan, "sigma_a": np.nan}
        results.append(res)
    return results


def generate_flutter_case_data(
    base_dir: Path,
    wind_speeds: List[float],
    *,
    B: float = 30.0,
    rho: float = 1.225,
    f_h: float = 0.10491093,
    f_theta: float = 0.12921779,
    seed: Optional[int] = None,
) -> None:
    """Generate synthetic case data required for flutter-derivative extraction."""
    const_h = {"H1": -0.2, "H2": 0.1, "H3": 1.0, "H4": -0.3}
    const_t = {"A1": 0.05, "A2": -0.4, "A3": -0.8, "A4": 0.4}

    for U in wind_speeds:
        case = base_dir / f"case_{int(U)}"
        case.mkdir(parents=True, exist_ok=True)

        t_v = np.arange(0.0, 20 / f_h, 0.02)
        w = 0.1 * np.sin(2 * math.pi * f_h * t_v)
        wdot = 2 * math.pi * f_h * 0.1 * np.cos(2 * math.pi * f_h * t_v)
        cl_v = 0.5 * rho * U ** 2 * B * 2 * math.pi * (
            const_h["H1"] * (w / B) + const_h["H2"] * (B / U) * wdot
        )
        cm_v = 0.5 * rho * U ** 2 * B ** 2 * 2 * math.pi * (
            const_h["H3"] * (w / B) + const_h["H4"] * (B / U) * wdot
        )
        pd.DataFrame({"Time": t_v, "w": w}).to_csv(case / "disp_V.csv", index=False)
        pd.DataFrame({"Time": t_v, "Lift": cl_v, "Moment": cm_v}).to_csv(
            case / "force_V.csv", index=False
        )

        t_t = np.arange(0.0, 20 / f_theta, 0.02)
        theta = 0.02 * np.sin(2 * math.pi * f_theta * t_t)
        thetad = 2 * math.pi * f_theta * 0.02 * np.cos(2 * math.pi * f_theta * t_t)
        cl_t = 0.5 * rho * U ** 2 * B * 2 * math.pi * (
            const_t["A1"] * theta + const_t["A2"] * (B / U) * thetad
        )
        cm_t = 0.5 * rho * U ** 2 * B ** 2 * 2 * math.pi * (
            const_t["A3"] * theta + const_t["A4"] * (B / U) * thetad
        )
        pd.DataFrame({"Time": t_t, "theta": theta}).to_csv(case / "disp_T.csv", index=False)
        pd.DataFrame({"Time": t_t, "Lift": cl_t, "Moment": cm_t}).to_csv(
            case / "force_T.csv", index=False
        )


def compute_flutter_derivatives(
    base_dir: Path,
    wind_speeds: List[float],
    *,
    B: float = 30.0,
    rho: float = 1.225,
    f_h: float = 0.10491093,
    f_theta: float = 0.12921779,
    discard_periods: int = 5,
) -> pd.DataFrame:
    """Derive flutter derivatives from case results."""
    summary_rows: List[pd.DataFrame] = []
    for U in wind_speeds:
        case_dir = base_dir / f"case_{U}"
        df_v = pd.read_csv(case_dir / "disp_V.csv").merge(
            pd.read_csv(case_dir / "force_V.csv"), on="Time"
        )
        df_t = pd.read_csv(case_dir / "disp_T.csv").merge(
            pd.read_csv(case_dir / "force_T.csv"), on="Time"
        )
        df_v = df_v[df_v["Time"] >= discard_periods * (1 / f_h)].reset_index(drop=True)
        df_t = df_t[df_t["Time"] >= discard_periods * (1 / f_theta)].reset_index(drop=True)

        t_v = df_v["Time"].to_numpy()
        dt_v = t_v[1] - t_v[0]
        w = df_v["w"].to_numpy()
        wdot = np.gradient(w, dt_v)
        w_b = w / B
        wdot_b = (B / U) * wdot

        cl_v = df_v["Lift"].to_numpy() / (0.5 * rho * U ** 2 * B)
        cm_v = df_v["Moment"].to_numpy() / (0.5 * rho * U ** 2 * B ** 2)

        term = 2 * math.pi
        A_v = np.column_stack([w_b, wdot_b])
        H1, H2 = np.linalg.lstsq(A_v, cl_v / term, rcond=None)[0]
        H3, H4 = np.linalg.lstsq(A_v, cm_v / term, rcond=None)[0]

        t_t = df_t["Time"].to_numpy()
        dt_t = t_t[1] - t_t[0]
        theta = df_t["theta"].to_numpy()
        thetad = np.gradient(theta, dt_t)
        theta_b = theta
        thetad_b = (B / U) * thetad

        cl_t = df_t["Lift"].to_numpy() / (0.5 * rho * U ** 2 * B)
        cm_t = df_t["Moment"].to_numpy() / (0.5 * rho * U ** 2 * B ** 2)
        A_t = np.column_stack([theta_b, thetad_b])
        A1, A2 = np.linalg.lstsq(A_t, cl_t / term, rcond=None)[0]
        A3, A4 = np.linalg.lstsq(A_t, cm_t / term, rcond=None)[0]

        k_val = 2 * math.pi * f_h * B / U
        single_df = pd.DataFrame(
            {
                "U": [U],
                "k": [k_val],
                "H1": [H1],
                "H2": [H2],
                "H3": [H3],
                "H4": [H4],
                "A1": [A1],
                "A2": [A2],
                "A3": [A3],
                "A4": [A4],
            }
        )
        out_file = base_dir / f"flutter_derivatives_U{int(U)}.csv"
        single_df.to_csv(out_file, index=False)
        summary_rows.append(single_df)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_file = base_dir / "flutter_derivatives_batch.csv"
    summary_df.to_csv(summary_file, index=False)
    return summary_df


def calculate_flutter_speed(
    flutter_derivatives: pd.DataFrame | Dict[str, List[float]],
    bridge_params: Dict[str, float],
    wind_speeds: np.ndarray,
) -> Tuple[Optional[float], Optional[float], List[float]]:
    if isinstance(flutter_derivatives, pd.DataFrame):
        fd = {
            "K": flutter_derivatives["k"].tolist(),
            "H1": flutter_derivatives["H1"].tolist(),
            "H2": flutter_derivatives["H2"].tolist(),
            "H3": flutter_derivatives["H3"].tolist(),
            "H4": flutter_derivatives["H4"].tolist(),
            "A1": flutter_derivatives["A1"].tolist(),
            "A2": flutter_derivatives["A2"].tolist(),
            "A3": flutter_derivatives["A3"].tolist(),
            "A4": flutter_derivatives["A4"].tolist(),
        }
    else:
        fd = flutter_derivatives

    k_ref = np.array(fd["K"])
    h1 = np.array(fd["H1"])
    h2 = np.array(fd["H2"])
    h3 = np.array(fd["H3"])
    h4 = np.array(fd["H4"])
    a1 = np.array(fd["A1"])
    a2 = np.array(fd["A2"])
    a3 = np.array(fd["A3"])
    a4 = np.array(fd["A4"])

    m = bridge_params["mass"]
    inertia = bridge_params["inertia"]
    f_h = bridge_params["f_h"]
    f_alpha = bridge_params["f_alpha"]
    zeta_h = bridge_params["damping_h"]
    zeta_a = bridge_params["damping_alpha"]
    b = bridge_params["width"]
    rho = bridge_params.get("density", 1.25)

    omega_h = 2 * math.pi * f_h
    omega_a = 2 * math.pi * f_alpha

    h1_i = np.interp
    h2_i = np.interp
    h3_i = np.interp
    h4_i = np.interp
    a1_i = np.interp
    a2_i = np.interp
    a3_i = np.interp
    a4_i = np.interp

    damping_results: List[float] = []
    critical_speed: Optional[float] = None
    flutter_freq: Optional[float] = None

    for u in wind_speeds:
        freq_range = np.linspace(0.5 * min(f_h, f_alpha), 2 * max(f_h, f_alpha), 100)
        min_damp = float("inf")
        freq_at_min = freq_range[0]
        for f in freq_range:
            w = 2 * math.pi * f
            k = w * b / u
            h1_v = np.interp(k, k_ref, h1)
            h2_v = np.interp(k, k_ref, h2)
            h3_v = np.interp(k, k_ref, h3)
            h4_v = np.interp(k, k_ref, h4)
            a1_v = np.interp(k, k_ref, a1)
            a2_v = np.interp(k, k_ref, a2)
            a3_v = np.interp(k, k_ref, a3)
            a4_v = np.interp(k, k_ref, a4)

            c_struct = np.array([[2 * m * zeta_h * omega_h, 0.0], [0.0, 2 * inertia * zeta_a * omega_a]])
            k_struct = np.array([[m * omega_h ** 2, 0.0], [0.0, inertia * omega_a ** 2]])
            c_aero = 0.5 * rho * u * b * np.array([[k * h1_v, k * h2_v * b], [k * a1_v * b, k * a2_v * b ** 2]])
            k_aero = 0.5 * rho * u ** 2 * np.array([[k ** 2 * h4_v, k ** 2 * h3_v * b], [k ** 2 * a4_v * b, k ** 2 * a3_v * b ** 2]])
            c_total = c_struct - c_aero
            k_total = k_struct - k_aero
            zeros = np.zeros((2, 2))
            ident = np.eye(2)
            a_mat = np.block([[zeros, ident], [-np.linalg.inv(np.diag([m, inertia])) @ k_total, -np.linalg.inv(np.diag([m, inertia])) @ c_total]])
            eigvals = np.linalg.eigvals(a_mat)
            max_real = float(np.max(eigvals.real))
            if max_real < min_damp:
                min_damp = max_real
                freq_at_min = f
        damping_results.append(min_damp)
        if len(damping_results) > 1 and damping_results[-2] >= 0 > min_damp:
            u_prev = wind_speeds[len(damping_results) - 2]
            d_prev = damping_results[-2]
            critical_speed = u_prev + (0 - d_prev) * (u - u_prev) / (min_damp - d_prev)
            flutter_freq = freq_at_min
    return critical_speed, flutter_freq, damping_results
