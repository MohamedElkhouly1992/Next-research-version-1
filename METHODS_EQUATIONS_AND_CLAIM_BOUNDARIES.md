# S4 Maintenance 4.0 — implemented research equations

## Digital Twin state model
Normalized component degradation state: x_i = D_i / D_crit,i.

x_{t+1} = clip[x_t + r_i F_i(u_t,w_t) Δt, 0, x_max]

The UKF performs measurement correction:

x_hat(t) = x_minus(t) + K_t [y_t - h(x_minus(t))].

## Probabilistic RUL
Each Monte Carlo trajectory samples literature-prior degradation rates and bootstrapped recent operating stress.
RUL_i^(k) = inf{h : x_i(t+h) >= 1}.
The software reports p05, p50, p95 and P(critical within 30/90/365 d).

## S4 prescriptive optimization
Decision vector:
z = [T_zone,set, T_CHWS,set, alpha_fan, alpha_pump, maintenance_delay, maintenance_action].

Four Pareto objectives are minimized simultaneously:
1. Energy cost over the planning horizon.
2. Occupancy-weighted comparative discomfort proxy.
3. Predicted equipment-health / critical-state risk.
4. Maintenance + downtime + expected failure-risk cost.

Weights are used only after optimization to select a representative compromise from the Pareto archive.

## S0–S4 policies
S0 reactive: near-critical threshold.
S1 scheduled: fixed calendar intervals.
S2 condition-based: current degradation threshold.
S3 adaptive: stress-adjusted threshold plus first-order projection/RUL trigger.
S4 Maintenance 4.0: UKF-updated digital-twin state + probabilistic risk + MOAPO joint control/maintenance prescription.

## Claim discipline
The comfort model in v1 is a comparative screening proxy, not a substitute for a fully parameterized ISO 7730 PMV/PPD calculation.
The exact stress coefficients, degradation priors, costs and service restoration fractions are exposed as study assumptions and should be calibrated before operational deployment.
