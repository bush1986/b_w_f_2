from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

from config import Settings
from sampling import define_random_variables
from fsi import (
    FSISimulationManager,
    generate_flutter_case_data,
    compute_flutter_derivatives,
    calculate_flutter_speed,
)
from reliability import create_normalized_limit_state, iterate_until_convergence
from fragility import build_all_capacities, fit_fragility_curve, compute_annual_pf


logger = logging.getLogger(__name__)


def run(settings: Settings, *, use_fsi: bool = True) -> None:
    random_vars = settings.random_vars
    mu, std, dists = define_random_variables(**random_vars)
    pop_size = settings.nsga_pop
    n_gen = settings.nsga_gen
    seed = settings.random_seed
    sampling_cfg = settings.sampling
    thresholds = settings.thresholds
    flutter_cfg = settings.flutter
    fd_base = Path(flutter_cfg.get("case_dir", "flutter_cases"))
    fd_speeds = flutter_cfg.get("wind_speeds", [20, 25, 30, 35, 40, 45, 50, 55, 60])
    fsi_manager = None
    if use_fsi and settings.use_fsi:
        try:
            fsi_manager = FSISimulationManager(settings.wbpj_template, settings.working_dir)
        except Exception as exc:  # pragma: no cover - external dependency
            logger.error("FSI管理器初始化失败: %s", exc)
            use_fsi = False

    if not (fd_base / "flutter_derivatives_batch.csv").exists():
        generate_flutter_case_data(fd_base, fd_speeds, seed=seed)
    flutter_df = compute_flutter_derivatives(fd_base, fd_speeds)
    speed_range = np.linspace(min(fd_speeds), max(fd_speeds), 80)
    u_cr, f_flut, _ = calculate_flutter_speed(flutter_df, settings.bridge_params, speed_range)
    if u_cr:
        logger.info("颤振临界风速: %.2f m/s, 频率 %.2f Hz", u_cr, f_flut)
    else:
        logger.info("给定范围内未找到颤振临界风速")

    g_func = create_normalized_limit_state()
    rsm, history = iterate_until_convergence(
        fsi_manager,
        mu,
        std,
        dists,
        g_func=g_func,
        max_iter=settings.convergence.get("max_iter", 15),
        tol=settings.convergence.get("tol", 0.01),
        beta_tol=settings.convergence.get("beta_tol", 0.005),
        pop_size=pop_size,
        n_gen=n_gen,
        use_fsi=use_fsi,
        seed=seed,
        sampling_cfg=sampling_cfg,
    )
    capacity_dict = build_all_capacities(dists, rsm, thresholds, seed=seed)
    results = fit_fragility_curve(capacity_dict)
    print("\n悬索桥抗风易损性分析结果")
    print("=" * 85)
    hdr = f"{'损伤状态':<20}{'θ':<12}{'β':<12}{'SE':<10}{'KS':<10}{'年失效概率':<15}{'50年失效概率'}"
    print(hdr)
    print("-" * 85)
    for label, (theta, beta, se_beta, ks) in results.items():
        pf = compute_annual_pf(dists, capacity_dict[label], seed=seed)
        pf50 = 1 - (1 - pf) ** 50 if not np.isnan(pf) else np.nan
        print(f"{label:<20}{theta:<12.4f}{beta:<12.4f}{se_beta:<10.4f}{ks:<10.4f}{pf:<15.6f}{pf50:.6f}")
    if fsi_manager:
        fsi_manager.cleanup()
