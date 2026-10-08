# Relationship to the previous Mohsen research engines

This S4 package is a **standalone higher-level extension** of the earlier research software, not a replacement for it.

## Earlier MOAPO–CatBoost engine

The previous engine focused on surrogate-assisted Pareto optimization of HVAC operation using four objectives. S4 retains the Pareto/MOAPO decision philosophy but changes the problem from static control optimization to **joint control + maintenance prescription**. The S4 decision vector therefore adds maintenance timing and maintenance action.

## Earlier degradation-economics engine

The previous Paper 2 engine quantified degradation-associated excess energy and maintenance economics using a counterfactual healthy baseline. S4 retains the economic logic but adds **state estimation, future risk and online prescription**.

## New S4 layer

The new chain is:

`BMS / DesignBuilder data -> Digital Twin UKF -> component health states -> probabilistic RUL/risk -> MOAPO prescription -> S0–S4 lifecycle comparison`

The software can use DesignBuilder-derived or field/BMS datasets directly through its column-mapping interface. For a future v2, CatBoost residual learning can be inserted between the physics prediction and UKF measurement correction when enough labelled operating data become available.
