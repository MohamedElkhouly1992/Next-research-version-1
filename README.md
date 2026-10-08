# S4 Maintenance 4.0 Research Software v1

## Research purpose

This software extends the Mohsen degradation-aware HVAC framework from S0–S3 to an S4 Maintenance 4.0 policy.

The implemented hierarchy is:

- **S0 — Reactive:** near-critical intervention.
- **S1 — Scheduled:** fixed calendar maintenance.
- **S2 — Condition-Based:** current-state degradation threshold.
- **S3 — Stress-Adaptive / forecast-assisted:** stress-adjusted threshold plus short-horizon projection/RUL trigger.
- **S4 — Maintenance 4.0:** measurement-updated Digital Twin + probabilistic RUL/risk + MOAPO joint supervisory-control and maintenance prescription.

## Core modules

1. Data ingestion and column mapping for CSV/XLSX DesignBuilder/BMS datasets.
2. Five-component physics-informed Digital Twin for compressor/chiller, coil, filter, fan and pump.
3. Unscented Kalman Filter (UKF) for online health-state correction when condition-sensitive measurements are available.
4. Probabilistic RUL by Monte Carlo propagation of literature-prior rate uncertainty and recent operating stress.
5. MOAPO Pareto prescriptive optimization of:
   - zone cooling setpoint,
   - CHWS setpoint,
   - fan command,
   - pump command,
   - maintenance delay,
   - maintenance action.
6. S0–S4 lifecycle comparison.
7. Automatic manuscript tables, 300-dpi figures, intervention logs, equations and reproducibility manifest.

## Digital Twin states

The state vector is normalized by the component critical state:

`x = [D_comp/Dcrit_comp, D_coil/Dcrit_coil, D_filter/Dcrit_filter, D_fan/Dcrit_fan, D_pump/Dcrit_pump]`

`x_i = 1` corresponds to the study critical threshold.

The model uses literature-prior annual degradation rates from the current research framework, exposed as editable assumptions rather than hidden constants.

## Measurement channels

The UKF can assimilate:

- chiller COP → primarily informs compressor/chiller state,
- coil UA ratio → coil fouling state,
- filter differential pressure → filter loading,
- fan efficiency ratio → fan state,
- pump efficiency ratio → pump state.

A component without a mapped measurement remains mainly **open-loop physics-informed**. The software reports sensor coverage explicitly so such states are not mislabeled as field-validated.

## Probabilistic RUL

For every component the software reports:

- 5th percentile RUL,
- median RUL,
- 95th percentile RUL,
- probability of reaching critical condition within 30 days,
- within 90 days,
- within 365 days.

The v1 distribution reflects degradation-prior uncertainty and observed/recent stress. It should not be presented as site-calibrated failure prognosis until independent aging/failure data are available.

## S4 Pareto objectives

S4 minimizes four non-collapsed objectives:

1. Energy cost.
2. Comparative comfort/discomfort proxy.
3. Equipment health / probability-of-critical-state risk.
4. Maintenance + downtime + expected failure-risk cost.

Objective weights are **not** used during Pareto optimization. They are used only after optimization to select a representative balanced compromise from the non-dominated archive.

## Important physical interpretation

The fan/pump command benefit uses a variable-speed affinity-law screening approximation. If the installed FCUs are constant-fan, the resulting fan savings represent a retrofit/control scenario rather than as-is hardware performance.

The v1 comfort module is intentionally a comparative PPD-like screening proxy. For a journal manuscript making strong comfort claims, replace it with the complete ISO 7730 / ASHRAE 55 PMV/PPD calculation using defensible met, clo, RH, MRT and air-speed inputs.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

or on Windows double-click `run_local.bat`.

## Numerical demo

```bash
python s4_maintenance40_engine.py --demo --output_dir demo_results
```

## Main automatic outputs

- `T01_digital_twin_state_history.csv`
- `T02_sensor_observability.csv`
- `T03_probabilistic_RUL_summary.csv`
- `T04_RUL_monte_carlo_samples.csv`
- `T05_S4_MOAPO_Pareto_archive.csv`
- `T06_S4_representative_prescriptions.csv`
- `T07_S4_MOAPO_convergence.csv`
- `T08_S0_S4_strategy_summary.csv`
- `T09_S0_S4_strategy_timeseries.csv`
- `T10_S0_S4_intervention_log.csv`
- manuscript figures at 300 dpi
- `METHODS_EQUATIONS_AND_CLAIM_BOUNDARIES.md`
- `MANUSCRIPT_RUN_SUMMARY.md`
- `run_manifest.json`

## Recommended publication validation sequence

1. Verify the UKF on synthetic fault/degradation trajectories where true hidden state is known.
2. Use fault-injected EnergyPlus/DesignBuilder cases to test state recovery and false alarms.
3. Validate component condition channels on independent public chiller/AHU/RTU datasets.
4. Collect real BMS condition measurements and before/after-maintenance recovery data.
5. Calibrate degradation-rate priors and restoration factors to the site.
6. Only then present the RUL layer as site-specific prognosis.
