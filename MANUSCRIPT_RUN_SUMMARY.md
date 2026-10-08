# Manuscript-ready run summary

Digital Twin final normalized states:
| component   |     state |
|:------------|----------:|
| compressor  | 0.0990102 |
| coil        | 0.0993386 |
| filter      | 0.106901  |
| fan         | 0.13053   |
| pump        | 0.136911  |

Probabilistic RUL:
| component   |   current_state_fraction_of_critical |   current_health_pct |   RUL_p05_days |   RUL_p50_days |   RUL_p95_days |   P_critical_30d |   P_critical_90d |   P_critical_365d |   censored_at_max_horizon_pct |
|:------------|-------------------------------------:|---------------------:|---------------:|---------------:|---------------:|-----------------:|-----------------:|------------------:|------------------------------:|
| compressor  |                            0.0990102 |              90.099  |            inf |            inf |          inf   |                0 |        0         |          0        |                           100 |
| coil        |                            0.0993386 |              90.0661 |            inf |            inf |          inf   |                0 |        0         |          0        |                           100 |
| filter      |                            0.106901  |              89.3099 |             88 |            156 |          324.6 |                0 |        0.0666667 |          0.991667 |                             0 |
| fan         |                            0.13053   |              86.947  |            inf |            inf |          inf   |                0 |        0         |          0        |                           100 |
| pump        |                            0.136911  |              86.3089 |            inf |            inf |          inf   |                0 |        0         |          0        |                           100 |

S0–S4 lifecycle comparison:
| strategy   |   years |   total_energy_MWh |   annualized_energy_MWh |   total_cost |   annualized_cost |   mean_PPD_proxy |   maintenance_and_replacement_actions |   replacements |   downtime_h |   final_mean_degradation |   final_max_degradation |   CO2_tonnes |
|:-----------|--------:|-------------------:|------------------------:|-------------:|------------------:|-----------------:|--------------------------------------:|---------------:|-------------:|-------------------------:|------------------------:|-------------:|
| S0         |    0.35 |            90.5686 |                 258.768 |     10868.2  |           31052.1 |          7.3698  |                                     0 |              0 |            0 |                 0.212905 |                0.527435 |      40.7559 |
| S1         |    0.35 |            89.9932 |                 257.123 |     10959.2  |           31311.9 |          7.02472 |                                     1 |              0 |            2 |                 0.134802 |                0.163732 |      40.4969 |
| S2         |    0.35 |            90.5686 |                 258.768 |     10868.2  |           31052.1 |          7.3698  |                                     0 |              0 |            0 |                 0.212905 |                0.527435 |      40.7559 |
| S3         |    0.35 |            76.6339 |                 218.954 |      9196.07 |           26274.5 |          9.79062 |                                     0 |              0 |            0 |                 0.200374 |                0.474869 |      34.4852 |
| S4         |    0.35 |            69.3679 |                 198.194 |      8644.15 |           24697.6 |         12.9842  |                                     2 |              0 |            4 |                 0.139859 |                0.179189 |      31.2156 |

For this configured scenario, S4 changed annualized HVAC energy by **23.41%** relative to S0 and annualized lifecycle cost by **20.46%** relative to S0. These values are scenario outputs, not transferable field claims.
