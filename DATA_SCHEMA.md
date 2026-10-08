# Data schema for S4 Maintenance 4.0 Research Software v1

The software uses manual column mapping, so exact input names are not mandatory. The following fields are recommended.

| Purpose | Recommended column | Unit / type | Required? |
|---|---|---|---|
| Time | timestamp | datetime | Recommended |
| Weather | outdoor_temp_C | °C | Recommended |
| Load | load_fraction | 0–1.25 | Recommended; otherwise inferred from energy |
| Occupancy | occupancy | count or normalized proxy | Recommended |
| Total HVAC energy | hvac_energy_kWh | kWh/timestep | Required unless component energies sum to total |
| Cooling energy | cooling_energy_kWh | kWh/timestep | Recommended |
| Fan energy | fan_energy_kWh | kWh/timestep | Recommended |
| Pump energy | pump_energy_kWh | kWh/timestep | Recommended |
| Auxiliary energy | aux_energy_kWh | kWh/timestep | Optional |
| Zone control | zone_setpoint_C | °C | Recommended |
| Chilled-water control | chw_setpoint_C | °C | Recommended if available |
| Fan command | fan_command | fraction | Recommended for variable-speed scenario |
| Pump command | pump_command | fraction | Recommended |
| Chiller condition | chiller_COP | dimensionless | Strongly recommended for measurement-updated compressor state |
| Coil condition | coil_UA_ratio | current UA / healthy UA | Strongly recommended for measurement-updated coil state |
| Filter condition | filter_dp_Pa | Pa | Strongly recommended for measurement-updated filter state |
| Fan condition | fan_eff_ratio | current / healthy efficiency | Optional but recommended |
| Pump condition | pump_eff_ratio | current / healthy efficiency | Optional but recommended |
| Service history | maintenance_component | compressor/coil/filter/fan/pump | Optional but valuable |

## Observability rule

A component state is only strongly measurement-informed when a corresponding condition-sensitive measurement is available. Missing channels are propagated using the physics-informed degradation prior and operating-stress model and are labelled accordingly in `T02_sensor_observability.csv`.

## Recommended real BMS data additions

For a stronger journal study, also archive raw CHWS/CHWR temperature, chilled-water flow, fan speed, pump speed, supply/return air temperature, filter differential pressure, chiller electric power, cooling load, valve positions, and maintenance work-order timestamps. These can be used later to replace proxy signals with directly derived physical indicators.
