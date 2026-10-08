from __future__ import annotations

import argparse
import io
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import norm
except Exception:
    norm = None


COMPONENTS = ("compressor", "coil", "filter", "fan", "pump")


@dataclass
class TwinConfig:
    # Normalized state x_i = physical degradation / critical degradation. x_i = 1 is critical.
    critical_physical: Dict[str, float] = field(default_factory=lambda: {
        "compressor": 0.25,
        "coil": 0.12,
        "filter": 1.00,
        "fan": 0.08,
        "pump": 0.17,
    })
    # Literature-prior physical annual rates (low, mode, high); user-editable in UI.
    annual_rate_prior: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        "compressor": (0.0025, 0.0100, 0.0140),
        "coil": (0.0025, 0.0050, 0.0100),
        "filter": (0.50, 1.00, 4.00),
        "fan": (0.0020, 0.0035, 0.0050),
        "pump": (0.0080, 0.0105, 0.0130),
    })
    irreversible_fraction: Dict[str, float] = field(default_factory=lambda: {
        "compressor": 0.75, "coil": 0.15, "filter": 0.0, "fan": 0.50, "pump": 0.50
    })
    remaining_reversible_after_service: Dict[str, float] = field(default_factory=lambda: {
        "compressor": 0.25, "coil": 0.05, "filter": 0.03, "fan": 0.15, "pump": 0.15
    })
    severity_factor: float = 1.0
    nominal_cop: float = 4.5
    clean_filter_dp_pa: float = 150.0
    final_filter_dp_ratio: float = 3.0
    coil_ua_shape: float = 1.0
    coil_energy_coupling: float = 0.50
    filter_system_pressure_share: float = 0.20
    process_noise_sd: float = 0.003
    initial_state_sd: float = 0.08
    measurement_rel_sd: float = 0.03
    max_state: float = 1.50


@dataclass
class EconomicConfig:
    electricity_tariff: float = 0.12
    emission_factor_kg_per_kwh: float = 0.45
    maintenance_cost: Dict[str, float] = field(default_factory=lambda: {
        "compressor": 1200.0, "coil": 500.0, "filter": 120.0, "fan": 450.0, "pump": 550.0
    })
    downtime_cost: Dict[str, float] = field(default_factory=lambda: {
        "compressor": 900.0, "coil": 250.0, "filter": 40.0, "fan": 250.0, "pump": 300.0
    })
    failure_cost: Dict[str, float] = field(default_factory=lambda: {
        "compressor": 12000.0, "coil": 3000.0, "filter": 800.0, "fan": 3500.0, "pump": 4500.0
    })


@dataclass
class ControlConfig:
    zone_setpoint_bounds: Tuple[float, float] = (22.0, 26.0)
    chw_setpoint_bounds: Tuple[float, float] = (5.0, 9.0)
    fan_command_bounds: Tuple[float, float] = (0.50, 1.00)
    pump_command_bounds: Tuple[float, float] = (0.50, 1.00)
    reference_zone_setpoint: float = 24.0
    reference_chw_setpoint: float = 6.0
    reference_fan_command: float = 1.0
    reference_pump_command: float = 1.0
    cooling_setpoint_sensitivity_per_c: float = 0.030
    chw_energy_sensitivity_per_c: float = 0.015
    cooling_share: float = 0.65
    fan_share: float = 0.18
    pump_share: float = 0.10
    aux_share: float = 0.07
    comfort_ppd_limit: float = 15.0


@dataclass
class OptimizationConfig:
    population: int = 18
    iterations: int = 10
    archive_size: int = 80
    seed: int = 42
    horizon_days: int = 30
    decision_interval_days: int = 14
    max_maintenance_delay_days: int = 30
    risk_limit: float = 0.10
    # post-Pareto compromise weights only; not used to collapse optimization objectives
    compromise_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)


@dataclass
class StrategyConfig:
    s0_threshold: float = 0.90
    s2_threshold: float = 0.70
    s3_base_threshold: float = 0.70
    hysteresis_fraction: float = 0.90
    s3_forecast_days: int = 30
    s3_rul_trigger_days: int = 30
    scheduled_intervals_days: Dict[str, int] = field(default_factory=lambda: {
        "compressor": 365, "coil": 180, "filter": 90, "fan": 365, "pump": 365
    })
    min_service_dwell_days: Dict[str, int] = field(default_factory=lambda: {
        "compressor": 30, "coil": 30, "filter": 21, "fan": 30, "pump": 30
    })


@dataclass
class ColumnMap:
    timestamp: Optional[str] = "timestamp"
    outdoor_temp_c: Optional[str] = "outdoor_temp_C"
    load_fraction: Optional[str] = "load_fraction"
    occupancy: Optional[str] = "occupancy"
    total_energy_kwh: Optional[str] = "hvac_energy_kWh"
    cooling_energy_kwh: Optional[str] = "cooling_energy_kWh"
    fan_energy_kwh: Optional[str] = "fan_energy_kWh"
    pump_energy_kwh: Optional[str] = "pump_energy_kWh"
    aux_energy_kwh: Optional[str] = "aux_energy_kWh"
    zone_setpoint_c: Optional[str] = "zone_setpoint_C"
    chw_setpoint_c: Optional[str] = "chw_setpoint_C"
    fan_command: Optional[str] = "fan_command"
    pump_command: Optional[str] = "pump_command"
    chiller_cop: Optional[str] = "chiller_COP"
    coil_ua_ratio: Optional[str] = "coil_UA_ratio"
    filter_dp_pa: Optional[str] = "filter_dp_Pa"
    fan_eff_ratio: Optional[str] = "fan_eff_ratio"
    pump_eff_ratio: Optional[str] = "pump_eff_ratio"
    maintenance_component: Optional[str] = "maintenance_component"


# ---------- Data ----------
def read_uploaded(files: List[Tuple[str, bytes]]) -> pd.DataFrame:
    frames = []
    for name, content in files:
        suffix = Path(name).suffix.lower()
        if suffix == ".csv":
            try:
                df = pd.read_csv(io.BytesIO(content))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
            df["source_file"] = name
            frames.append(df)
        elif suffix in {".xlsx", ".xls"}:
            xl = pd.ExcelFile(io.BytesIO(content))
            for sh in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sh)
                if len(df):
                    df["source_file"] = name
                    df["source_sheet"] = sh
                    frames.append(df)
    if not frames:
        raise ValueError("No readable CSV/XLSX input files.")
    return pd.concat(frames, ignore_index=True, sort=False)


def infer_timestep_hours(df: pd.DataFrame, timestamp_col: Optional[str]) -> float:
    if timestamp_col and timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna().sort_values()
        if len(ts) > 2:
            d = ts.diff().dropna().dt.total_seconds() / 3600.0
            d = d[(d > 0) & (d <= 24 * 31)]
            if len(d):
                return float(d.median())
    return 24.0


def num(df: pd.DataFrame, col: Optional[str], default=0.0) -> np.ndarray:
    if not col or col not in df.columns:
        return np.full(len(df), float(default))
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(float)


def clean_profile(df: pd.DataFrame, cmap: ColumnMap, ctrl: ControlConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if cmap.timestamp and cmap.timestamp in df.columns:
        out["timestamp"] = pd.to_datetime(df[cmap.timestamp], errors="coerce")
    else:
        out["timestamp"] = pd.date_range("2025-01-01", periods=len(df), freq="D")
    out["outdoor_temp_C"] = num(df, cmap.outdoor_temp_c, 30.0)
    load = num(df, cmap.load_fraction, np.nan)
    if np.isnan(load).all():
        # fallback load proxy from energy
        e = num(df, cmap.total_energy_kwh, 1.0)
        q = np.nanquantile(e, 0.99) if np.isfinite(e).any() else 1.0
        load = np.clip(e / max(q, 1e-9), 0, 1.25)
    else:
        med = np.nanmedian(load) if np.isfinite(load).any() else 0.5
        load = np.nan_to_num(load, nan=med)
    out["load_fraction"] = np.clip(load, 0, 1.25)
    out["occupancy"] = num(df, cmap.occupancy, 1.0)
    out["zone_setpoint_C"] = num(df, cmap.zone_setpoint_c, ctrl.reference_zone_setpoint)
    out["chw_setpoint_C"] = num(df, cmap.chw_setpoint_c, ctrl.reference_chw_setpoint)
    out["fan_command"] = np.clip(num(df, cmap.fan_command, ctrl.reference_fan_command), 0.05, 1.5)
    out["pump_command"] = np.clip(num(df, cmap.pump_command, ctrl.reference_pump_command), 0.05, 1.5)

    total = num(df, cmap.total_energy_kwh, np.nan)
    cool = num(df, cmap.cooling_energy_kwh, np.nan)
    fan = num(df, cmap.fan_energy_kwh, np.nan)
    pump = num(df, cmap.pump_energy_kwh, np.nan)
    aux = num(df, cmap.aux_energy_kwh, np.nan)
    if np.isnan(total).all():
        parts = np.vstack([cool, fan, pump, aux])
        total = np.nansum(parts, axis=0)
    if np.isnan(cool).all(): cool = total * ctrl.cooling_share
    if np.isnan(fan).all(): fan = total * ctrl.fan_share
    if np.isnan(pump).all(): pump = total * ctrl.pump_share
    if np.isnan(aux).all(): aux = total * ctrl.aux_share
    out["hvac_energy_kWh"] = np.nan_to_num(total, nan=0.0)
    out["cooling_energy_kWh"] = np.nan_to_num(cool, nan=0.0)
    out["fan_energy_kWh"] = np.nan_to_num(fan, nan=0.0)
    out["pump_energy_kWh"] = np.nan_to_num(pump, nan=0.0)
    out["aux_energy_kWh"] = np.nan_to_num(aux, nan=0.0)

    # Observation channels; NaN means unavailable and the UKF will remain open-loop for that state.
    for key, col in [
        ("chiller_COP", cmap.chiller_cop), ("coil_UA_ratio", cmap.coil_ua_ratio),
        ("filter_dp_Pa", cmap.filter_dp_pa), ("fan_eff_ratio", cmap.fan_eff_ratio),
        ("pump_eff_ratio", cmap.pump_eff_ratio)
    ]:
        if col and col in df.columns:
            out[key] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[key] = np.nan
    if cmap.maintenance_component and cmap.maintenance_component in df.columns:
        out["maintenance_component"] = df[cmap.maintenance_component].fillna("").astype(str)
    else:
        out["maintenance_component"] = ""
    return out


# ---------- Physics ----------
def normalized_mode_rates(cfg: TwinConfig) -> np.ndarray:
    return np.array([cfg.annual_rate_prior[c][1] / cfg.critical_physical[c] for c in COMPONENTS], float)


def stress_vector(row: Dict[str, float]) -> np.ndarray:
    L = float(np.clip(row.get("load_fraction", 0.5), 0, 1.25))
    tout = float(row.get("outdoor_temp_C", 30.0))
    chw = float(row.get("chw_setpoint_C", 6.0))
    fan = float(np.clip(row.get("fan_command", 1.0), 0.05, 1.5))
    pump = float(np.clip(row.get("pump_command", 1.0), 0.05, 1.5))
    lift = np.clip((tout - chw - 18.0) / 18.0, 0.0, 1.5)
    return np.array([
        0.55 + 0.75*L + 0.35*lift,                    # compressor
        0.60 + 0.65*L + 0.25*fan,                     # coil
        0.45 + 0.75*fan + 0.10*max(tout-30.0, 0)/10, # filter
        0.50 + 0.70*fan**2 + 0.15*L,                  # fan
        0.50 + 0.70*pump**2 + 0.15*L,                 # pump
    ], float)


def transition(x: np.ndarray, row: Dict[str, float], dt_hours: float, cfg: TwinConfig,
               rate_multiplier: Optional[np.ndarray] = None) -> np.ndarray:
    rates = normalized_mode_rates(cfg)
    if rate_multiplier is not None:
        rates = rates * np.asarray(rate_multiplier, float)
    dx = rates * cfg.severity_factor * stress_vector(row) * (dt_hours / 8766.0)
    return np.clip(np.asarray(x, float) + dx, 0.0, cfg.max_state)


def physical_signals(x: np.ndarray, cfg: TwinConfig) -> Dict[str, float]:
    x = np.asarray(x, float)
    comp, coil, filt, fan, pump = x
    # physical degradation = normalized state * component critical state
    dcomp = comp * cfg.critical_physical["compressor"]
    dfan = fan * cfg.critical_physical["fan"]
    dpump = pump * cfg.critical_physical["pump"]
    cop_ratio = max(0.25, 1.0 - dcomp)
    ua_ratio = 1.0 / (1.0 + cfg.coil_ua_shape * max(coil, 0.0))
    filter_ratio = 1.0 + (cfg.final_filter_dp_ratio - 1.0) * max(filt, 0.0) ** 1.5
    fan_eff = max(0.40, 1.0 - dfan)
    pump_eff = max(0.40, 1.0 - dpump)
    return {
        "chiller_COP": cfg.nominal_cop * cop_ratio,
        "coil_UA_ratio": ua_ratio,
        "filter_dp_Pa": cfg.clean_filter_dp_pa * filter_ratio,
        "fan_eff_ratio": fan_eff,
        "pump_eff_ratio": pump_eff,
    }


def degradation_energy_multipliers(x: np.ndarray, cfg: TwinConfig) -> Tuple[float, float, float]:
    sig = physical_signals(x, cfg)
    cooling = (cfg.nominal_cop / max(sig["chiller_COP"], 0.2)) * (1.0 + cfg.coil_energy_coupling * (1.0 - sig["coil_UA_ratio"]))
    filt_sys = (1.0 - cfg.filter_system_pressure_share) + cfg.filter_system_pressure_share * (sig["filter_dp_Pa"] / cfg.clean_filter_dp_pa)
    fan = filt_sys / max(sig["fan_eff_ratio"], 0.2)
    pump = 1.0 / max(sig["pump_eff_ratio"], 0.2)
    return float(cooling), float(fan), float(pump)


def apply_service(x: np.ndarray, component: str, cfg: TwinConfig) -> np.ndarray:
    y = np.asarray(x, float).copy()
    if component not in COMPONENTS:
        return y
    i = COMPONENTS.index(component)
    irr = float(cfg.irreversible_fraction[component])
    rho = float(cfg.remaining_reversible_after_service[component])
    # Approximate decomposition: irreversible fraction of current accumulated degradation remains;
    # reversible portion is reduced to rho of its pre-service level.
    y[i] = np.clip(y[i] * (irr + (1.0 - irr) * rho), 0.0, cfg.max_state)
    return y


def replacement_reset(x: np.ndarray, component: str, cfg: TwinConfig) -> np.ndarray:
    y = np.asarray(x, float).copy()
    if component in COMPONENTS:
        y[COMPONENTS.index(component)] = 0.02
    return y


# ---------- UKF ----------
class UKF:
    def __init__(self, n: int, x0: np.ndarray, P0: np.ndarray, alpha=0.15, beta=2.0, kappa=0.0):
        self.n = n
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.lam = alpha**2 * (n + kappa) - n
        self.Wm = np.full(2*n + 1, 1.0 / (2*(n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        n = self.n
        Pj = P.copy()
        for _ in range(6):
            try:
                S = np.linalg.cholesky((n + self.lam) * Pj)
                break
            except np.linalg.LinAlgError:
                Pj = Pj + np.eye(n) * 1e-8
        else:
            S = np.linalg.cholesky((n + self.lam) * (Pj + np.eye(n)*1e-5))
        pts = [x]
        for i in range(n):
            pts.append(x + S[:, i])
            pts.append(x - S[:, i])
        return np.asarray(pts)

    def predict(self, f, Q: np.ndarray):
        sig = self.sigma_points(self.x, self.P)
        Y = np.asarray([f(s) for s in sig])
        xm = np.sum(self.Wm[:, None] * Y, axis=0)
        Pm = Q.copy()
        for i in range(len(Y)):
            d = (Y[i] - xm)[:, None]
            Pm += self.Wc[i] * (d @ d.T)
        self.x, self.P = xm, Pm

    def update(self, z: np.ndarray, h, R: np.ndarray):
        sig = self.sigma_points(self.x, self.P)
        Z = np.asarray([h(s) for s in sig])
        zm = np.sum(self.Wm[:, None] * Z, axis=0)
        S = R.copy()
        C = np.zeros((self.n, len(z)))
        for i in range(len(sig)):
            dz = (Z[i] - zm)[:, None]
            dx = (sig[i] - self.x)[:, None]
            S += self.Wc[i] * (dz @ dz.T)
            C += self.Wc[i] * (dx @ dz.T)
        try:
            K = C @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = C @ np.linalg.pinv(S)
        innov = z - zm
        self.x = self.x + K @ innov
        self.P = self.P - K @ S @ K.T
        self.P = 0.5*(self.P + self.P.T) + np.eye(self.n)*1e-10
        return innov


def run_digital_twin(profile: pd.DataFrame, cfg: TwinConfig, dt_hours: float,
                     initial_state: Optional[Sequence[float]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    x0 = np.array(initial_state if initial_state is not None else [0.05]*5, float)
    P0 = np.eye(5) * cfg.initial_state_sd**2
    ukf = UKF(5, x0, P0)
    Q = np.eye(5) * cfg.process_noise_sd**2
    obs_fields = ["chiller_COP", "coil_UA_ratio", "filter_dp_Pa", "fan_eff_ratio", "pump_eff_ratio"]
    rows = []
    observed_counts = {k: 0 for k in obs_fields}

    for i, r in profile.reset_index(drop=True).iterrows():
        row = r.to_dict()
        ukf.predict(lambda x: transition(x, row, dt_hours, cfg), Q)

        # Apply observed maintenance event before assimilating post-service measurements.
        comp = str(row.get("maintenance_component", "")).strip().lower()
        if comp in COMPONENTS:
            ukf.x = apply_service(ukf.x, comp, cfg)
            ukf.P += np.eye(5)*0.01

        available = [k for k in obs_fields if pd.notna(row.get(k, np.nan))]
        innovation_norm = np.nan
        if available:
            z = np.array([float(row[k]) for k in available])
            def h(x):
                sig = physical_signals(x, cfg)
                return np.array([sig[k] for k in available], float)
            pred = h(ukf.x)
            # Relative measurement noise with a small absolute floor.
            sd = np.maximum(np.abs(pred) * cfg.measurement_rel_sd, np.array([0.02 if "ratio" in k else (2.0 if "dp" in k else 0.05) for k in available]))
            innov = ukf.update(z, h, np.diag(sd**2))
            innovation_norm = float(np.linalg.norm(innov / np.maximum(sd, 1e-9)))
            for k in available: observed_counts[k] += 1

        ukf.x = np.clip(ukf.x, 0, cfg.max_state)
        diag_sd = np.sqrt(np.maximum(np.diag(ukf.P), 0))
        sig = physical_signals(ukf.x, cfg)
        rec = {"timestamp": row["timestamp"], "innovation_norm": innovation_norm}
        for j, c in enumerate(COMPONENTS):
            rec[f"{c}_state"] = ukf.x[j]
            rec[f"{c}_state_sd"] = diag_sd[j]
            rec[f"{c}_health_pct"] = 100*max(0.0, 1.0 - ukf.x[j])
        rec.update({f"pred_{k}": v for k, v in sig.items()})
        rows.append(rec)

    hist = pd.DataFrame(rows)
    observability = pd.DataFrame({
        "measurement": obs_fields,
        "observed_rows": [observed_counts[k] for k in obs_fields],
        "coverage_pct": [100*observed_counts[k]/max(len(profile),1) for k in obs_fields],
        "primarily_informs_component": list(COMPONENTS),
    })
    meta = {
        "final_state": ukf.x.copy(), "final_cov": ukf.P.copy(),
        "observability": observability,
        "warning": "States without direct measurement coverage are mainly open-loop physics-informed estimates and should not be presented as field-validated health estimates."
    }
    return hist, meta


# ---------- RUL ----------
def recent_stress_rows(profile: pd.DataFrame, days: int, dt_hours: float) -> pd.DataFrame:
    n = max(1, int(round(days * 24 / max(dt_hours, 1e-9))))
    return profile.tail(min(n, len(profile))).reset_index(drop=True)


def sample_rate_multiplier(rng: np.random.Generator, cfg: TwinConfig) -> np.ndarray:
    vals = []
    for c in COMPONENTS:
        lo, mode, hi = cfg.annual_rate_prior[c]
        draw = rng.triangular(lo, mode, hi)
        vals.append(draw / max(mode, 1e-12))
    return np.asarray(vals)


def probabilistic_rul(current_state: np.ndarray, profile: pd.DataFrame, cfg: TwinConfig, dt_hours: float,
                      n_mc: int = 500, max_days: int = 1825, seed: int = 42,
                      future_window_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    future_pool = recent_stress_rows(profile, max(30, future_window_days), dt_hours)
    if future_pool.empty:
        raise ValueError("No operating profile available for RUL propagation.")
    step_days = dt_hours / 24.0
    n_steps = int(math.ceil(max_days / step_days))
    n_mc = int(n_mc)
    states = np.repeat(np.asarray(current_state, float)[None, :], n_mc, axis=0)
    rul = np.full((n_mc, 5), np.inf)
    initial_failed = states >= 1.0
    rul[initial_failed] = 0.0

    # Sample uncertain annual-rate multipliers once per trajectory.
    rate_mult = np.ones((n_mc, 5), float)
    for j,c in enumerate(COMPONENTS):
        lo, mode, hi = cfg.annual_rate_prior[c]
        rate_mult[:,j] = rng.triangular(lo, mode, hi, size=n_mc) / max(mode, 1e-12)
    rates = normalized_mode_rates(cfg)[None,:] * cfg.severity_factor * rate_mult

    # Extract recent operating pool into numeric arrays to avoid slow pandas row access inside MC loops.
    Lpool = np.clip(pd.to_numeric(future_pool['load_fraction'], errors='coerce').fillna(0.5).to_numpy(float),0,1.25)
    Tpool = pd.to_numeric(future_pool['outdoor_temp_C'], errors='coerce').fillna(30).to_numpy(float)
    Cpool = pd.to_numeric(future_pool['chw_setpoint_C'], errors='coerce').fillna(6).to_numpy(float)
    Fpool = np.clip(pd.to_numeric(future_pool['fan_command'], errors='coerce').fillna(1).to_numpy(float),0.05,1.5)
    Ppool = np.clip(pd.to_numeric(future_pool['pump_command'], errors='coerce').fillna(1).to_numpy(float),0.05,1.5)
    dt_year = dt_hours / 8766.0

    for k in range(n_steps):
        idx = rng.integers(0, len(future_pool), size=n_mc)
        L=Lpool[idx]; tout=Tpool[idx]; chw=Cpool[idx]; fan=Fpool[idx]; pump=Ppool[idx]
        lift=np.clip((tout-chw-18.0)/18.0,0,1.5)
        stress=np.column_stack([
            0.55+0.75*L+0.35*lift,
            0.60+0.65*L+0.25*fan,
            0.45+0.75*fan+0.10*np.maximum(tout-30.0,0)/10,
            0.50+0.70*fan**2+0.15*L,
            0.50+0.70*pump**2+0.15*L,
        ])
        states=np.clip(states+rates*stress*dt_year,0,cfg.max_state)
        newly=(states>=1.0)&np.isinf(rul)
        rul[newly]=(k+1)*step_days
        if np.all(np.isfinite(rul)):
            break

    rows=[]
    for j,c in enumerate(COMPONENTS):
        vals=rul[:,j]; censored=~np.isfinite(vals); finite=vals[np.isfinite(vals)]
        q=lambda p: float(np.quantile(finite,p)) if len(finite) else np.inf
        rows.append({
            'component':c,
            'current_state_fraction_of_critical':float(current_state[j]),
            'current_health_pct':float(100*max(0,1-current_state[j])),
            'RUL_p05_days':q(0.05),'RUL_p50_days':q(0.50),'RUL_p95_days':q(0.95),
            'P_critical_30d':float(np.mean(vals<=30)),'P_critical_90d':float(np.mean(vals<=90)),'P_critical_365d':float(np.mean(vals<=365)),
            'censored_at_max_horizon_pct':float(100*np.mean(censored)),
        })
    summary=pd.DataFrame(rows)
    capped=np.where(np.isfinite(rul),rul,max_days+step_days)
    long=pd.DataFrame({
        'mc_run':np.repeat(np.arange(1,n_mc+1),5),
        'component':np.tile(np.array(COMPONENTS,dtype=object),n_mc),
        'RUL_days':capped.reshape(-1),
    })
    return summary,long


def approximate_failure_probability(x: np.ndarray, rows: pd.DataFrame, cfg: TwinConfig, dt_hours: float, horizon_days: int) -> np.ndarray:
    # Fast moment approximation for optimization: deterministic projection plus uncertainty margin.
    if rows.empty:
        stress = np.ones(5)
    else:
        stress = np.mean(np.vstack([stress_vector(r.to_dict()) for _,r in rows.iterrows()]), axis=0)
    rates = normalized_mode_rates(cfg) * cfg.severity_factor * stress
    mu = x + rates * (horizon_days / 365.25)
    # Relative uncertainty combines prior width and process uncertainty.
    prior_rel = []
    for c in COMPONENTS:
        lo,mode,hi = cfg.annual_rate_prior[c]
        prior_rel.append(max((hi-lo)/(4*max(mode,1e-12)),0.08))
    sd = np.maximum(0.015, (mu - x) * np.asarray(prior_rel) + cfg.process_noise_sd*np.sqrt(max(horizon_days,1)))
    z = (1.0 - mu) / np.maximum(sd,1e-9)
    if norm is not None:
        p = 1.0 - norm.cdf(z)
    else:
        # logistic approximation to Gaussian tail
        p = 1.0 / (1.0 + np.exp(1.702*z))
    return np.clip(p, 0, 1)


# ---------- Energy / comfort ----------
def control_and_degradation_energy(base: Dict[str,float], x: np.ndarray, controls: Sequence[float], twin: TwinConfig, ctrl: ControlConfig) -> Dict[str,float]:
    zone, chw, fan_cmd, pump_cmd = map(float, controls)
    deg_cool, deg_fan, deg_pump = degradation_energy_multipliers(x, twin)
    zmod = max(0.75, 1.0 - ctrl.cooling_setpoint_sensitivity_per_c*(zone-ctrl.reference_zone_setpoint))
    chwmod = max(0.80, 1.0 - ctrl.chw_energy_sensitivity_per_c*(chw-ctrl.reference_chw_setpoint))
    cool = max(0, base["cooling"] * deg_cool * zmod * chwmod)
    fan = max(0, base["fan"] * deg_fan * (fan_cmd/max(ctrl.reference_fan_command,1e-9))**3)
    pump = max(0, base["pump"] * deg_pump * (pump_cmd/max(ctrl.reference_pump_command,1e-9))**3)
    aux = max(0, base["aux"])
    return {"cooling":cool,"fan":fan,"pump":pump,"aux":aux,"total":cool+fan+pump+aux}


def comfort_ppd_proxy(row: Dict[str,float], x: np.ndarray, controls: Sequence[float], ctrl: ControlConfig) -> float:
    zone, chw, fan_cmd, pump_cmd = map(float, controls)
    occ = max(float(row.get("occupancy",1.0)),0.0)
    if occ <= 0:
        return 0.0
    # Comparative screening proxy; explicitly not field-validated PMV/PPD.
    ppd = 6.0 + 5.0*abs(zone-24.0) + 22.0*max(0.0,0.75-fan_cmd) + 2.0*max(0.0,chw-7.0)
    ppd += 4.0*float(x[1]) + 3.0*float(x[2])
    return float(np.clip(ppd, 5.0, 100.0))


def base_energy_from_row(row: Dict[str,float], ctrl: ControlConfig) -> Dict[str,float]:
    total = max(float(row.get("hvac_energy_kWh",0.0)),0.0)
    cool = float(row.get("cooling_energy_kWh",np.nan))
    fan = float(row.get("fan_energy_kWh",np.nan))
    pump = float(row.get("pump_energy_kWh",np.nan))
    aux = float(row.get("aux_energy_kWh",np.nan))
    if not np.isfinite(cool): cool = total*ctrl.cooling_share
    if not np.isfinite(fan): fan = total*ctrl.fan_share
    if not np.isfinite(pump): pump = total*ctrl.pump_share
    if not np.isfinite(aux): aux = total*ctrl.aux_share
    s = cool+fan+pump+aux
    if s <= 0 and total > 0:
        cool,fan,pump,aux = total*ctrl.cooling_share,total*ctrl.fan_share,total*ctrl.pump_share,total*ctrl.aux_share
    return {"cooling":cool,"fan":fan,"pump":pump,"aux":aux}


# ---------- Pareto utilities ----------
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    n = len(F); keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]: continue
        for j in range(n):
            if i != j and dominates(F[j], F[i]):
                keep[i] = False; break
    return keep


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n,m = F.shape
    if n == 0: return np.array([])
    if n <= 2: return np.full(n,np.inf)
    d = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:,j])
        d[order[0]] = d[order[-1]] = np.inf
        lo,hi = F[order[0],j],F[order[-1],j]
        if hi <= lo: continue
        for k in range(1,n-1):
            d[order[k]] += (F[order[k+1],j]-F[order[k-1],j])/(hi-lo)
    return d


def update_archive(Xa: np.ndarray, Fa: np.ndarray, X: np.ndarray, F: np.ndarray, max_size: int) -> Tuple[np.ndarray,np.ndarray]:
    XX = np.vstack([Xa,X]) if len(Xa) else X.copy()
    FF = np.vstack([Fa,F]) if len(Fa) else F.copy()
    # rounded duplicate objective removal
    _, idx = np.unique(np.round(FF,10), axis=0, return_index=True)
    XX,FF = XX[idx],FF[idx]
    keep = nondominated_mask(FF)
    XX,FF = XX[keep],FF[keep]
    while len(XX) > max_size:
        cd = crowding_distance(FF)
        finite = np.where(np.isfinite(cd), cd, np.inf)
        rm = int(np.argmin(finite))
        XX,FF = np.delete(XX,rm,0),np.delete(FF,rm,0)
    return XX,FF


def repair(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.clip(X, bounds[:,0], bounds[:,1])


class PrescriptiveObjective:
    # Decision vector: zone_sp, chw_sp, fan_cmd, pump_cmd, maintenance_delay_days, action_code
    def __init__(self, state: np.ndarray, profile: pd.DataFrame, twin: TwinConfig, econ: EconomicConfig,
                 ctrl: ControlConfig, opt: OptimizationConfig, dt_hours: float):
        self.state=np.asarray(state,float); self.profile=profile.reset_index(drop=True); self.twin=twin; self.econ=econ; self.ctrl=ctrl; self.opt=opt; self.dt_hours=dt_hours
        self.bounds=np.array([
            ctrl.zone_setpoint_bounds, ctrl.chw_setpoint_bounds, ctrl.fan_command_bounds, ctrl.pump_command_bounds,
            (0.0,float(opt.max_maintenance_delay_days)), (0.0,5.49)
        ],float)
        self._mcost=np.array([0.0]+[econ.maintenance_cost[c]+econ.downtime_cost[c] for c in COMPONENTS],float)
        self._fcost=np.array([econ.failure_cost[c] for c in COMPONENTS],float)
        self._rates=normalized_mode_rates(twin)*twin.severity_factor
        self._prior_rel=np.array([max((twin.annual_rate_prior[c][2]-twin.annual_rate_prior[c][0])/(4*max(twin.annual_rate_prior[c][1],1e-12)),0.08) for c in COMPONENTS],float)

    def decode_action(self, v: float) -> str:
        i=int(np.clip(np.rint(v),0,5))
        return "none" if i==0 else COMPONENTS[i-1]

    def _signals_vec(self, states: np.ndarray):
        comp,coil,filt,fan,pump=[states[:,j] for j in range(5)]
        dcomp=comp*self.twin.critical_physical['compressor']; dfan=fan*self.twin.critical_physical['fan']; dpump=pump*self.twin.critical_physical['pump']
        cop_ratio=np.maximum(0.25,1.0-dcomp)
        ua_ratio=1.0/(1.0+self.twin.coil_ua_shape*np.maximum(coil,0))
        filter_ratio=1.0+(self.twin.final_filter_dp_ratio-1.0)*np.maximum(filt,0)**1.5
        fan_eff=np.maximum(0.40,1.0-dfan); pump_eff=np.maximum(0.40,1.0-dpump)
        cool=(1.0/cop_ratio)*(1.0+self.twin.coil_energy_coupling*(1.0-ua_ratio))
        filt_sys=(1.0-self.twin.filter_system_pressure_share)+self.twin.filter_system_pressure_share*filter_ratio
        fan_mult=filt_sys/fan_eff; pump_mult=1.0/pump_eff
        return cool,fan_mult,pump_mult

    def _stress_vec(self, row: Dict[str,float], fan_cmd: np.ndarray, pump_cmd: np.ndarray, chw: np.ndarray):
        L=float(np.clip(row.get('load_fraction',0.5),0,1.25)); tout=float(row.get('outdoor_temp_C',30.0))
        lift=np.clip((tout-chw-18.0)/18.0,0,1.5)
        return np.column_stack([
            0.55+0.75*L+0.35*lift,
            0.60+0.65*L+0.25*fan_cmd,
            0.45+0.75*fan_cmd+0.10*max(tout-30.0,0)/10,
            0.50+0.70*fan_cmd**2+0.15*L,
            0.50+0.70*pump_cmd**2+0.15*L,
        ])

    def evaluate_one(self, z: np.ndarray) -> np.ndarray:
        return self.evaluate(np.asarray(z,float)[None,:])[0]

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X=np.asarray(X,float)
        if X.ndim==1:X=X[None,:]
        n=len(X); zone=X[:,0]; chw=X[:,1]; fan_cmd=X[:,2]; pump_cmd=X[:,3]
        delay=np.rint(X[:,4]).astype(int); action=np.clip(np.rint(X[:,5]).astype(int),0,5)
        states=np.repeat(self.state[None,:],n,axis=0)
        H=max(1,int(self.opt.horizon_days)); rows=self.profile.tail(max(1,min(len(self.profile),H))).reset_index(drop=True)
        ecost=np.zeros(n); csum=np.zeros(n); osum=np.zeros(n)
        maintained=np.zeros(n,bool)
        for d in range(H):
            r=rows.iloc[d%len(rows)].to_dict()
            do=(action>0)&(~maintained)&(d>=delay)
            if np.any(do):
                for ai in range(1,6):
                    mask=do&(action==ai)
                    if np.any(mask):
                        c=COMPONENTS[ai-1]; irr=self.twin.irreversible_fraction[c]; rho=self.twin.remaining_reversible_after_service[c]
                        states[mask,ai-1]=np.clip(states[mask,ai-1]*(irr+(1-irr)*rho),0,self.twin.max_state)
                maintained[do]=True
            cool_m,fan_m,pump_m=self._signals_vec(states)
            base=base_energy_from_row(r,self.ctrl)
            zmod=np.maximum(0.75,1.0-self.ctrl.cooling_setpoint_sensitivity_per_c*(zone-self.ctrl.reference_zone_setpoint))
            chwmod=np.maximum(0.80,1.0-self.ctrl.chw_energy_sensitivity_per_c*(chw-self.ctrl.reference_chw_setpoint))
            energy=base['cooling']*cool_m*zmod*chwmod + base['fan']*fan_m*(fan_cmd/max(self.ctrl.reference_fan_command,1e-9))**3 + base['pump']*pump_m*(pump_cmd/max(self.ctrl.reference_pump_command,1e-9))**3 + base['aux']
            ecost += energy*self.econ.electricity_tariff
            occ=max(float(r.get('occupancy',1.0)),0.0); w=max(occ,1e-6)
            ppd=6.0+5.0*np.abs(zone-24.0)+22.0*np.maximum(0,0.75-fan_cmd)+2.0*np.maximum(0,chw-7.0)+4.0*states[:,1]+3.0*states[:,2]
            ppd=np.clip(ppd,5,100); csum+=ppd*w; osum+=w
            stress=self._stress_vec(r,fan_cmd,pump_cmd,chw)
            states=np.clip(states+self._rates[None,:]*stress*(1.0/365.25),0,self.twin.max_state)
        mean_ppd=csum/np.maximum(osum,1e-12)
        # Candidate-specific risk using final-day stress as local projection approximation.
        ravg=rows.iloc[-1].to_dict(); stress=self._stress_vec(ravg,fan_cmd,pump_cmd,chw)
        mu=states+self._rates[None,:]*stress*(H/365.25)
        sd=np.maximum(0.015,(mu-states)*self._prior_rel[None,:]+self.twin.process_noise_sd*np.sqrt(max(H,1)))
        z=(1.0-mu)/np.maximum(sd,1e-9)
        if norm is not None:risk=1.0-norm.cdf(z)
        else:risk=1.0/(1.0+np.exp(1.702*z))
        risk=np.clip(risk,0,1); hrisk=np.mean(risk,axis=1)
        failcost=risk@self._fcost; mecon=self._mcost[action]+failcost
        comfort_obj=np.where(mean_ppd>self.ctrl.comfort_ppd_limit,mean_ppd+10*(mean_ppd-self.ctrl.comfort_ppd_limit),mean_ppd)
        maxrisk=np.max(risk,axis=1); hrisk_obj=hrisk+np.where(maxrisk>self.opt.risk_limit,5*(maxrisk-self.opt.risk_limit),0)
        return np.column_stack([ecost,comfort_obj,hrisk_obj,mecon])


def random_population(rng,n,bounds):
    return bounds[:,0]+rng.random((n,len(bounds)))*(bounds[:,1]-bounds[:,0])


def run_moapo(obj: PrescriptiveObjective, opt: OptimizationConfig, seed: Optional[int]=None) -> Dict[str,Any]:
    rng=np.random.default_rng(opt.seed if seed is None else seed)
    n,K=opt.population,opt.iterations; bounds=obj.bounds
    lo,hi=bounds[:,0],bounds[:,1]; span=hi-lo
    X=random_population(rng,n,bounds); F=obj.evaluate(X)
    Xa,Fa=update_archive(np.empty((0,X.shape[1])),np.empty((0,4)),X,F,opt.archive_size)
    history=[]
    for k in range(K):
        progress=k/max(K-1,1); p_ah=0.5*(1+math.cos(math.pi*progress)); fscale=0.5*(1-progress)+0.05
        # rank proxy: number of dominators
        rank=np.array([sum(dominates(F[j],F[i]) for j in range(n) if j!=i) for i in range(n)])
        order=np.argsort(rank); n_dr=max(1,int(math.ceil(n*(0.10+0.20*(1-progress)))))
        poor=set(order[-n_dr:].tolist()); good=set(order[:n_dr].tolist())
        cda=crowding_distance(Fa)
        finite=np.where(np.isfinite(cda),np.maximum(cda,1e-9),np.nanmax(cda[np.isfinite(cda)])*2 if np.any(np.isfinite(cda)) else 1.0)
        probs=finite/finite.sum() if finite.sum()>0 else np.ones(len(Fa))/len(Fa)
        Xnew=X.copy()
        for i in range(n):
            leader=Xa[rng.choice(len(Xa),p=probs)]
            if i in poor and rng.random()<0.55:
                cand=lo+rng.random(len(lo))*span
            elif i in good and rng.random()<0.50:
                mask=rng.random(len(lo))<max(1/len(lo),0.5*(1-progress));
                if not np.any(mask): mask[rng.integers(0,len(lo))]=True
                cand=X[i].copy(); noise=rng.normal(0,1,len(lo))*span*(0.08*(1-progress)+0.01)
                cand[mask]=0.5*X[i,mask]+0.5*leader[mask]+noise[mask]
            else:
                a,b=rng.choice(n,2,replace=False)
                if rng.random()<p_ah:
                    cand=X[i]+fscale*(X[a]-X[b])+rng.uniform(-0.25,0.25,len(lo))*span*(1-progress)
                else:
                    cand=X[i]+(0.8+0.6*rng.random())*fscale*(leader-X[i])+0.35*fscale*(X[a]-X[b])
            Xnew[i]=repair(cand[None,:],bounds)[0]
        Fnew=obj.evaluate(Xnew)
        for i in range(n):
            if dominates(Fnew[i],F[i]) or (not dominates(F[i],Fnew[i]) and rng.random()<(0.45+0.25*(1-progress))):
                X[i],F[i]=Xnew[i],Fnew[i]
        Xa,Fa=update_archive(Xa,Fa,X,F,opt.archive_size)
        if k==0 or (k+1)%max(1,K//10)==0 or k==K-1:
            history.append({"iteration":k+1,"archive_size":len(Fa),"best_energy_cost":float(np.min(Fa[:,0])),"best_risk":float(np.min(Fa[:,2]))})
    return {"X":Xa,"F":Fa,"history":pd.DataFrame(history)}


def pareto_table(result: Dict[str,Any], obj: PrescriptiveObjective) -> pd.DataFrame:
    X,F=result["X"],result["F"]
    rows=[]
    for i in range(len(X)):
        z=X[i]; f=F[i]
        rows.append({
            "solution_id":i+1,"zone_setpoint_C":z[0],"chw_setpoint_C":z[1],"fan_command":z[2],"pump_command":z[3],
            "maintenance_delay_days":z[4],"maintenance_action":obj.decode_action(z[5]),
            "energy_cost_objective":f[0],"comfort_ppd_objective":f[1],"health_risk_objective":f[2],"maintenance_risk_cost_objective":f[3]
        })
    return pd.DataFrame(rows)


def representatives(pareto: pd.DataFrame, weights=(0.25,0.25,0.25,0.25)) -> pd.DataFrame:
    if pareto.empty:return pareto
    cols=["energy_cost_objective","comfort_ppd_objective","health_risk_objective","maintenance_risk_cost_objective"]
    F=pareto[cols].to_numpy(float); lo=np.min(F,0); hi=np.max(F,0); Z=(F-lo)/(hi-lo+1e-12)
    labels={"energy_priority":int(np.argmin(F[:,0])),"comfort_priority":int(np.argmin(F[:,1])),"health_priority":int(np.argmin(F[:,2])),"cost_risk_priority":int(np.argmin(F[:,3]))}
    w=np.asarray(weights,float); w=w/max(w.sum(),1e-12); dist=np.sqrt(np.sum(w[None,:]*Z**2,axis=1)); labels["balanced_compromise"]=int(np.argmin(dist))
    rows=[]
    for lab,idx in labels.items():
        rec=pareto.iloc[idx].to_dict(); rec["representative"]=lab; rec["normalized_distance_to_utopia"]=float(dist[idx]); rows.append(rec)
    return pd.DataFrame(rows)


# ---------- S0-S4 comparison ----------
def strategy_controls(strategy: str, ctrl: ControlConfig) -> np.ndarray:
    if strategy in {"S3","S4"}:
        return np.array([24.5,6.5,0.80,0.80],float)
    return np.array([ctrl.reference_zone_setpoint,ctrl.reference_chw_setpoint,ctrl.reference_fan_command,ctrl.reference_pump_command],float)


def simulate_strategies(start_state: np.ndarray, profile: pd.DataFrame, twin: TwinConfig, econ: EconomicConfig,
                        ctrl: ControlConfig, opt: OptimizationConfig, scfg: StrategyConfig,
                        years: float=2.0, seed:int=42, s4_optimizer_scale:float=0.55) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    rng=np.random.default_rng(seed)
    steps=max(1,int(round(years*365.25)))
    prof=profile.reset_index(drop=True)
    summary=[]; histories=[]; interventions=[]
    for strategy in ["S0","S1","S2","S3","S4"]:
        x=np.asarray(start_state,float).copy(); controls=strategy_controls(strategy,ctrl)
        last_service={c:0 for c in COMPONENTS}; latch={c:False for c in COMPONENTS}
        energy=cost=comfort_num=occ_den=co2=0.0; actions=0; repl=0; downtime=0.0
        s4_plan=None; s4_plan_age=10**9
        for d in range(steps):
            r=prof.iloc[d%len(prof)].to_dict()
            # determine maintenance candidates
            service=[]
            if strategy=="S0":
                for j,c in enumerate(COMPONENTS):
                    if x[j]>=scfg.s0_threshold and not latch[c]: service.append(c); latch[c]=True
                    if x[j]<scfg.s0_threshold*scfg.hysteresis_fraction: latch[c]=False
            elif strategy=="S1":
                for c in COMPONENTS:
                    if d-last_service[c]>=scfg.scheduled_intervals_days[c]: service.append(c)
            elif strategy=="S2":
                for j,c in enumerate(COMPONENTS):
                    if x[j]>=scfg.s2_threshold and not latch[c]: service.append(c); latch[c]=True
                    if x[j]<scfg.s2_threshold*scfg.hysteresis_fraction: latch[c]=False
            elif strategy=="S3":
                st=stress_vector(r); rates=normalized_mode_rates(twin)*twin.severity_factor*st
                projected=x+rates*(scfg.s3_forecast_days/365.25)
                # stress-adjusted threshold: higher stress -> earlier intervention
                thr=np.clip(scfg.s3_base_threshold-0.06*(st-1.0),0.50,0.85)
                local_rul=np.divide(1-x,rates/365.25,out=np.full(5,np.inf),where=rates>1e-12)
                for j,c in enumerate(COMPONENTS):
                    demand=(x[j]>=thr[j]) or (projected[j]>=thr[j]) or (local_rul[j]<=scfg.s3_rul_trigger_days)
                    if demand and not latch[c]:service.append(c);latch[c]=True
                    if x[j]<thr[j]*scfg.hysteresis_fraction:latch[c]=False
            elif strategy=="S4":
                # Re-optimize weekly or when current plan expires. Use a reduced budget for lifecycle simulation.
                if s4_plan is None or s4_plan_age>=opt.decision_interval_days:
                    local_opt=OptimizationConfig(**asdict(opt))
                    local_opt.population=max(10,int(round(opt.population*s4_optimizer_scale)))
                    local_opt.iterations=max(8,int(round(opt.iterations*s4_optimizer_scale)))
                    local_opt.archive_size=min(opt.archive_size,60)
                    pobj=PrescriptiveObjective(x,prof.iloc[max(0,(d%len(prof))-30):min(len(prof),(d%len(prof))+60)] if len(prof)>2 else prof,twin,econ,ctrl,local_opt,24.0)
                    res=run_moapo(pobj,local_opt,seed+rng.integers(0,10**9))
                    reps=representatives(pareto_table(res,pobj),local_opt.compromise_weights)
                    s4_plan=reps.loc[reps["representative"]=="balanced_compromise"].iloc[0].to_dict()
                    s4_plan_age=0
                controls=np.array([s4_plan["zone_setpoint_C"],s4_plan["chw_setpoint_C"],s4_plan["fan_command"],s4_plan["pump_command"]],float)
                if s4_plan_age>=int(round(float(s4_plan["maintenance_delay_days"]))) and s4_plan["maintenance_action"]!="none":
                    service=[str(s4_plan["maintenance_action"])]
                    s4_plan=None
                else:
                    s4_plan_age+=1

            # Enforce a minimum service dwell to avoid implausible repeated interventions.
            service=[c for c in service if d-last_service[c] >= scfg.min_service_dwell_days[c]]

            # replacement takes priority at critical
            critical=[c for j,c in enumerate(COMPONENTS) if x[j]>=1.0]
            if critical:
                service=[]
                for c in critical:
                    before=float(x[COMPONENTS.index(c)]); x=replacement_reset(x,c,twin); repl+=1; actions+=1
                    cst=econ.failure_cost[c]; cost+=cst; downtime+=8.0
                    interventions.append({"strategy":strategy,"day":d,"component":c,"action":"replacement","state_before":before,"state_after":float(x[COMPONENTS.index(c)]),"cost":cst})
            else:
                for c in service:
                    before=float(x[COMPONENTS.index(c)]); x=apply_service(x,c,twin); last_service[c]=d; actions+=1
                    cst=econ.maintenance_cost[c]+econ.downtime_cost[c]; cost+=cst; downtime+=2.0
                    interventions.append({"strategy":strategy,"day":d,"component":c,"action":"maintenance","state_before":before,"state_after":float(x[COMPONENTS.index(c)]),"cost":cst})

            r["zone_setpoint_C"],r["chw_setpoint_C"],r["fan_command"],r["pump_command"]=controls
            ee=control_and_degradation_energy(base_energy_from_row(r,ctrl),x,controls,twin,ctrl)
            e=ee["total"]; e_cost=e*econ.electricity_tariff
            ppd=comfort_ppd_proxy(r,x,controls,ctrl); occ=max(float(r.get("occupancy",1.0)),0.0)
            energy+=e; cost+=e_cost; co2+=e*econ.emission_factor_kg_per_kwh; comfort_num+=ppd*max(occ,1e-6);occ_den+=max(occ,1e-6)
            risk=approximate_failure_probability(x,prof.tail(min(len(prof),30)),twin,24.0,30)
            histories.append({"strategy":strategy,"day":d,"energy_kWh":e,"cumulative_energy_kWh":energy,"cumulative_cost":cost,"PPD_proxy":ppd,
                              "mean_state":float(np.mean(x)),"max_state":float(np.max(x)),"Pcrit30_max":float(np.max(risk)),
                              **{f"{c}_state":float(x[j]) for j,c in enumerate(COMPONENTS)}})
            x=transition(x,r,24.0,twin)

        summary.append({"strategy":strategy,"years":years,"total_energy_MWh":energy/1000,"annualized_energy_MWh":energy/1000/years,
                        "total_cost":cost,"annualized_cost":cost/years,"mean_PPD_proxy":comfort_num/max(occ_den,1e-9),"maintenance_and_replacement_actions":actions,
                        "replacements":repl,"downtime_h":downtime,"final_mean_degradation":float(np.mean(x)),"final_max_degradation":float(np.max(x)),"CO2_tonnes":co2/1000})
    return pd.DataFrame(summary),pd.DataFrame(histories),pd.DataFrame(interventions)


# ---------- Figures / export ----------
def plot_results(twin_hist:pd.DataFrame,rul:pd.DataFrame,pareto:pd.DataFrame,strategy_summary:pd.DataFrame,strategy_hist:pd.DataFrame,outdir:Path):
    outdir.mkdir(parents=True,exist_ok=True)
    # DT states
    fig,ax=plt.subplots(figsize=(9,5))
    x=twin_hist["timestamp"] if "timestamp" in twin_hist else np.arange(len(twin_hist))
    for c in COMPONENTS: ax.plot(x,twin_hist[f"{c}_state"],label=c)
    ax.axhline(1.0,linestyle="--",linewidth=1,label="critical")
    ax.set_ylabel("Normalized degradation state (1 = critical)");ax.set_title("Digital Twin online health-state estimation");ax.legend(ncol=3);fig.tight_layout();fig.savefig(outdir/'Fig01_digital_twin_states.png',dpi=300);plt.close(fig)
    # RUL
    fig,ax=plt.subplots(figsize=(8,5)); y=np.arange(len(rul)); med=rul['RUL_p50_days'].replace(np.inf,np.nan).to_numpy(float); lo=rul['RUL_p05_days'].replace(np.inf,np.nan).to_numpy(float); hi=rul['RUL_p95_days'].replace(np.inf,np.nan).to_numpy(float)
    ax.errorbar(med,y,xerr=np.vstack([np.maximum(med-lo,0),np.maximum(hi-med,0)]),fmt='o',capsize=4);ax.set_yticks(y,labels=rul['component']);ax.set_xlabel('RUL (days)');ax.set_title('Probabilistic component RUL (median and 90% interval)');fig.tight_layout();fig.savefig(outdir/'Fig02_probabilistic_RUL.png',dpi=300);plt.close(fig)
    if len(pareto):
        fig,ax=plt.subplots(figsize=(7,5));sc=ax.scatter(pareto['energy_cost_objective'],pareto['comfort_ppd_objective'],c=pareto['health_risk_objective'],s=25+0.01*pareto['maintenance_risk_cost_objective'].clip(lower=0));ax.set_xlabel('Energy cost objective');ax.set_ylabel('Comfort PPD objective');ax.set_title('S4 MOAPO Pareto archive');fig.colorbar(sc,ax=ax,label='Health-risk objective');fig.tight_layout();fig.savefig(outdir/'Fig03_S4_MOAPO_Pareto.png',dpi=300);plt.close(fig)
    if len(strategy_summary):
        for metric,label,fname in [('annualized_energy_MWh','Annualized HVAC energy (MWh/y)','Fig04_S0_S4_energy.png'),('annualized_cost','Annualized lifecycle cost','Fig05_S0_S4_cost.png'),('maintenance_and_replacement_actions','Maintenance + replacement actions','Fig06_S0_S4_actions.png')]:
            fig,ax=plt.subplots(figsize=(7,4.5));ax.bar(strategy_summary['strategy'],strategy_summary[metric]);ax.set_ylabel(label);ax.set_title(f'S0–S4 comparison: {label}');fig.tight_layout();fig.savefig(outdir/fname,dpi=300);plt.close(fig)
    if len(strategy_hist):
        fig,ax=plt.subplots(figsize=(9,5));
        for s,g in strategy_hist.groupby('strategy'):ax.plot(g['day'],g['max_state'],label=s)
        ax.axhline(1,linestyle='--',linewidth=1);ax.set_xlabel('Simulation day');ax.set_ylabel('Maximum normalized degradation');ax.set_title('S0–S4 degradation trajectory');ax.legend();fig.tight_layout();fig.savefig(outdir/'Fig07_S0_S4_degradation_trajectory.png',dpi=300);plt.close(fig)


def write_package(results:Dict[str,Any],outdir:Path):
    outdir.mkdir(parents=True,exist_ok=True)
    results['twin_history'].to_csv(outdir/'T01_digital_twin_state_history.csv',index=False)
    results['observability'].to_csv(outdir/'T02_sensor_observability.csv',index=False)
    results['rul_summary'].to_csv(outdir/'T03_probabilistic_RUL_summary.csv',index=False)
    results['rul_samples'].to_csv(outdir/'T04_RUL_monte_carlo_samples.csv',index=False)
    results['pareto'].to_csv(outdir/'T05_S4_MOAPO_Pareto_archive.csv',index=False)
    results['representatives'].to_csv(outdir/'T06_S4_representative_prescriptions.csv',index=False)
    results['moapo_history'].to_csv(outdir/'T07_S4_MOAPO_convergence.csv',index=False)
    results['strategy_summary'].to_csv(outdir/'T08_S0_S4_strategy_summary.csv',index=False)
    results['strategy_history'].to_csv(outdir/'T09_S0_S4_strategy_timeseries.csv',index=False)
    results['interventions'].to_csv(outdir/'T10_S0_S4_intervention_log.csv',index=False)
    plot_results(results['twin_history'],results['rul_summary'],results['pareto'],results['strategy_summary'],results['strategy_history'],outdir/'figures')
    manifest={
        'software':'S4 Maintenance 4.0 Research Software','version':'1.0.0',
        'timestep_hours':results['dt_hours'],'twin_config':asdict(results['twin_config']),'economic_config':asdict(results['economic_config']),
        'control_config':asdict(results['control_config']),'optimization_config':asdict(results['optimization_config']),'strategy_config':asdict(results['strategy_config']),
        'column_map':asdict(results['column_map']),
        'claim_boundary':'UKF-updated states are measurement-informed only for mapped sensor channels. RUL is probabilistic scenario prognosis based on literature-prior rates and observed/recent stress until independently calibrated to aging/failure histories.'
    }
    (outdir/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    methods='''# S4 Maintenance 4.0 — implemented research equations\n\n## Digital Twin state model\nNormalized component degradation state: x_i = D_i / D_crit,i.\n\nx_{t+1} = clip[x_t + r_i F_i(u_t,w_t) Δt, 0, x_max]\n\nThe UKF performs measurement correction:\n\nx_hat(t) = x_minus(t) + K_t [y_t - h(x_minus(t))].\n\n## Probabilistic RUL\nEach Monte Carlo trajectory samples literature-prior degradation rates and bootstrapped recent operating stress.\nRUL_i^(k) = inf{h : x_i(t+h) >= 1}.\nThe software reports p05, p50, p95 and P(critical within 30/90/365 d).\n\n## S4 prescriptive optimization\nDecision vector:\nz = [T_zone,set, T_CHWS,set, alpha_fan, alpha_pump, maintenance_delay, maintenance_action].\n\nFour Pareto objectives are minimized simultaneously:\n1. Energy cost over the planning horizon.\n2. Occupancy-weighted comparative discomfort proxy.\n3. Predicted equipment-health / critical-state risk.\n4. Maintenance + downtime + expected failure-risk cost.\n\nWeights are used only after optimization to select a representative compromise from the Pareto archive.\n\n## S0–S4 policies\nS0 reactive: near-critical threshold.\nS1 scheduled: fixed calendar intervals.\nS2 condition-based: current degradation threshold.\nS3 adaptive: stress-adjusted threshold plus first-order projection/RUL trigger.\nS4 Maintenance 4.0: UKF-updated digital-twin state + probabilistic risk + MOAPO joint control/maintenance prescription.\n\n## Claim discipline\nThe comfort model in v1 is a comparative screening proxy, not a substitute for a fully parameterized ISO 7730 PMV/PPD calculation.\nThe exact stress coefficients, degradation priors, costs and service restoration fractions are exposed as study assumptions and should be calibrated before operational deployment.\n'''
    (outdir/'METHODS_EQUATIONS_AND_CLAIM_BOUNDARIES.md').write_text(methods,encoding='utf-8')
    s=results['strategy_summary'].set_index('strategy')
    if 'S4' in s.index and 'S0' in s.index:
        e_save=100*(s.loc['S0','annualized_energy_MWh']-s.loc['S4','annualized_energy_MWh'])/max(s.loc['S0','annualized_energy_MWh'],1e-9)
        c_save=100*(s.loc['S0','annualized_cost']-s.loc['S4','annualized_cost'])/max(abs(s.loc['S0','annualized_cost']),1e-9)
    else:e_save=c_save=np.nan
    summary=f'''# Manuscript-ready run summary\n\nDigital Twin final normalized states:\n{pd.DataFrame({'component':COMPONENTS,'state':results['final_state']}).to_markdown(index=False)}\n\nProbabilistic RUL:\n{results['rul_summary'].to_markdown(index=False)}\n\nS0–S4 lifecycle comparison:\n{results['strategy_summary'].to_markdown(index=False)}\n\nFor this configured scenario, S4 changed annualized HVAC energy by **{e_save:.2f}%** relative to S0 and annualized lifecycle cost by **{c_save:.2f}%** relative to S0. These values are scenario outputs, not transferable field claims.\n'''
    (outdir/'MANUSCRIPT_RUN_SUMMARY.md').write_text(summary,encoding='utf-8')


# ---------- Demo ----------
def synthetic_dataset(days:int=730,seed:int=42)->pd.DataFrame:
    rng=np.random.default_rng(seed); ts=pd.date_range('2024-01-01',periods=days,freq='D'); doy=ts.dayofyear.to_numpy()
    tout=29+7*np.sin(2*np.pi*(doy-170)/365)+rng.normal(0,1.2,days); load=np.clip(0.50+0.30*np.sin(2*np.pi*(doy-170)/365)+rng.normal(0,0.08,days),0.12,1.15);occ=np.where(ts.dayofweek<5,1.0,0.25)
    zone=np.full(days,24.0);chw=np.full(days,6.0);fan=np.ones(days);pump=np.ones(days)
    cfg=TwinConfig();x=np.array([0.04,0.03,0.08,0.03,0.03]);rows=[]
    for i in range(days):
        r={'load_fraction':load[i],'outdoor_temp_C':tout[i],'chw_setpoint_C':chw[i],'fan_command':fan[i],'pump_command':pump[i]}
        sig=physical_signals(x,cfg);dc,df,dp=degradation_energy_multipliers(x,cfg);healthy=480+620*load[i]+12*max(tout[i]-30,0);cool=healthy*0.65*dc;fe=healthy*0.18*df;pe=healthy*0.10*dp;aux=healthy*0.07;total=cool+fe+pe+aux+rng.normal(0,10)
        maint=''
        if i in {240,480}:maint='filter';x=apply_service(x,'filter',cfg)
        rows.append({'timestamp':ts[i],'outdoor_temp_C':tout[i],'load_fraction':load[i],'occupancy':occ[i],'zone_setpoint_C':zone[i],'chw_setpoint_C':chw[i],'fan_command':fan[i],'pump_command':pump[i],
                     'hvac_energy_kWh':max(total,1),'cooling_energy_kWh':max(cool,1),'fan_energy_kWh':max(fe,1),'pump_energy_kWh':max(pe,1),'aux_energy_kWh':max(aux,1),
                     'chiller_COP':sig['chiller_COP']*(1+rng.normal(0,0.012)),'coil_UA_ratio':sig['coil_UA_ratio']*(1+rng.normal(0,0.008)),'filter_dp_Pa':sig['filter_dp_Pa']*(1+rng.normal(0,0.015)),
                     'fan_eff_ratio':sig['fan_eff_ratio']*(1+rng.normal(0,0.006)),'pump_eff_ratio':sig['pump_eff_ratio']*(1+rng.normal(0,0.006)),'maintenance_component':maint})
        x=transition(x,r,24,cfg)
    return pd.DataFrame(rows)


def full_analysis(df:pd.DataFrame,cmap:ColumnMap,twin:TwinConfig,econ:EconomicConfig,ctrl:ControlConfig,opt:OptimizationConfig,scfg:StrategyConfig,
                  n_rul_mc:int=300,rul_max_days:int=1825,strategy_years:float=1.0)->Dict[str,Any]:
    profile=clean_profile(df,cmap,ctrl);dt=infer_timestep_hours(df,cmap.timestamp)
    twin_hist,meta=run_digital_twin(profile,twin,dt);state=np.asarray(meta['final_state'])
    rul,rul_samples=probabilistic_rul(state,profile,twin,dt,n_mc=n_rul_mc,max_days=rul_max_days,seed=opt.seed)
    pobj=PrescriptiveObjective(state,profile,twin,econ,ctrl,opt,dt);mores=run_moapo(pobj,opt);pareto=pareto_table(mores,pobj);reps=representatives(pareto,opt.compromise_weights)
    strat,shist,inter=simulate_strategies(state,profile,twin,econ,ctrl,opt,scfg,years=strategy_years,seed=opt.seed)
    return {'profile':profile,'dt_hours':dt,'twin_history':twin_hist,'observability':meta['observability'],'final_state':state,'final_cov':meta['final_cov'],
            'rul_summary':rul,'rul_samples':rul_samples,'pareto':pareto,'representatives':reps,'moapo_history':mores['history'],
            'strategy_summary':strat,'strategy_history':shist,'interventions':inter,'twin_config':twin,'economic_config':econ,'control_config':ctrl,'optimization_config':opt,'strategy_config':scfg,'column_map':cmap}


def demo_run(outdir:Path):
    df=synthetic_dataset(500); sample=outdir.parent/'sample_S4_Maintenance40_input.csv';df.to_csv(sample,index=False)
    res=full_analysis(df,ColumnMap(),TwinConfig(),EconomicConfig(),ControlConfig(),OptimizationConfig(population=16,iterations=12,archive_size=60),StrategyConfig(),n_rul_mc=120,rul_max_days=1000,strategy_years=0.35)
    write_package(res,outdir)
    return res


if __name__=='__main__':
    ap=argparse.ArgumentParser(description='S4 Maintenance 4.0 Research Software v1')
    ap.add_argument('--demo',action='store_true');ap.add_argument('--output_dir',default='S4_demo_results');args=ap.parse_args()
    if args.demo:
        out=Path(args.output_dir);demo_run(out);print(f'Demo completed: {out.resolve()}')
    else:
        print('Run streamlit_app.py for the interactive application or use --demo.')
