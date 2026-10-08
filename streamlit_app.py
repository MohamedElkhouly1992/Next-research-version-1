from __future__ import annotations
import io, tempfile, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from s4_maintenance40_engine import *

st.set_page_config(page_title="S4 Maintenance 4.0 Research Software",layout="wide",initial_sidebar_state="expanded")
st.title("S4 Maintenance 4.0 Research Software v1")
st.caption("Digital Twin state estimation • probabilistic RUL • MOAPO prescriptive maintenance • S0–S4 comparison")

with st.expander("Scientific scope and evidence boundary",expanded=False):
    st.markdown("""
This research platform extends the S0–S3 degradation-aware HVAC framework with an **S4 Maintenance 4.0** layer.

**S4 = measurement-updated Digital Twin + probabilistic prognosis + Pareto prescriptive control/maintenance + closed-loop service logic.**

The software intentionally distinguishes **measurement-informed states** from open-loop physics-informed states. A component without a mapped condition-sensitive measurement is not treated as field-validated merely because the state estimator produces a number. RUL is probabilistic but remains scenario-based until site-specific aging/failure histories are available.
""")

st.sidebar.header("1. Data")
mode=st.sidebar.radio("Input",["Built-in research demonstration","Upload CSV/XLSX"])
if mode=="Built-in research demonstration":
    df=synthetic_dataset(500);st.sidebar.success("500-day synthetic degradation/BMS dataset loaded")
else:
    up=st.sidebar.file_uploader("Upload one or more files",type=['csv','xlsx','xls'],accept_multiple_files=True)
    if not up:st.info("Upload HVAC/BMS or DesignBuilder-derived datasets to continue.");st.stop()
    try:df=read_uploaded([(f.name,f.getvalue()) for f in up])
    except Exception as e:st.exception(e);st.stop()

st.subheader("Input data audit")
a,b,c=st.columns(3);a.metric("Rows",f"{len(df):,}");b.metric("Columns",len(df.columns));c.metric("Sources",df['source_file'].nunique() if 'source_file' in df else 1)
st.dataframe(df.head(200),use_container_width=True)

cols=['<none>']+list(df.columns)
def pick(label,preferred,key):
    norm={str(c).lower():i for i,c in enumerate(cols)};idx=0
    for p in preferred:
        if p.lower() in norm:idx=norm[p.lower()];break
    v=st.sidebar.selectbox(label,cols,index=idx,key=key);return None if v=='<none>' else v

st.sidebar.header("2. Column mapping")
cmap=ColumnMap(
 timestamp=pick('Timestamp',['timestamp','datetime'],'m_ts'),outdoor_temp_c=pick('Outdoor temperature °C',['outdoor_temp_C','outdoor_drybulb_C'],'m_to'),load_fraction=pick('Normalized load',['load_fraction','load','normalized_load'],'m_load'),occupancy=pick('Occupancy/load proxy',['occupancy','people','occupied_flag'],'m_occ'),
 total_energy_kwh=pick('Total HVAC energy kWh/timestep',['hvac_energy_kWh','hvac_total_energy_kWh'],'m_e'),cooling_energy_kwh=pick('Cooling/chiller energy',['cooling_energy_kWh','chiller_energy_kWh'],'m_ec'),fan_energy_kwh=pick('Fan energy',['fan_energy_kWh'],'m_ef'),pump_energy_kwh=pick('Pump energy',['pump_energy_kWh'],'m_ep'),aux_energy_kwh=pick('Auxiliary energy',['aux_energy_kWh'],'m_ea'),
 zone_setpoint_c=pick('Zone setpoint °C',['zone_setpoint_C','candidate_zone_setpoint_C'],'m_zsp'),chw_setpoint_c=pick('CHWS setpoint °C',['chw_setpoint_C','chw_supply_setpoint_C'],'m_chw'),fan_command=pick('Fan command fraction',['fan_command','airflow_fraction','candidate_airflow_fraction'],'m_fc'),pump_command=pick('Pump command fraction',['pump_command','pump_speed_fraction'],'m_pc'),
 chiller_cop=pick('Measured chiller COP',['chiller_COP','COP_eff'],'m_cop'),coil_ua_ratio=pick('Measured/derived coil UA ratio',['coil_UA_ratio','ua_ratio'],'m_ua'),filter_dp_pa=pick('Filter differential pressure Pa',['filter_dp_Pa','filter_dp'],'m_fdp'),fan_eff_ratio=pick('Fan efficiency ratio',['fan_eff_ratio'],'m_fer'),pump_eff_ratio=pick('Pump efficiency ratio',['pump_eff_ratio'],'m_per'),maintenance_component=pick('Observed maintenance component',['maintenance_component'],'m_mc'))

st.sidebar.header("3. Digital Twin")
severity_name=st.sidebar.selectbox('Degradation severity prior',['Mild','Moderate','Severe','High'],index=1);sev={'Mild':0.8,'Moderate':1.0,'Severe':1.2,'High':1.5}[severity_name]
twin=TwinConfig(severity_factor=sev,nominal_cop=st.sidebar.number_input('Nominal healthy COP',2.0,8.0,4.5,0.1),clean_filter_dp_pa=st.sidebar.number_input('Clean filter ΔP (Pa)',10.0,1000.0,150.0,10.0),process_noise_sd=st.sidebar.number_input('UKF process-noise SD',0.0001,0.05,0.003,0.001,format='%.4f'),measurement_rel_sd=st.sidebar.number_input('Measurement relative SD',0.005,0.30,0.03,0.005))

st.sidebar.header("4. Economics")
econ=EconomicConfig(electricity_tariff=st.sidebar.number_input('Electricity tariff / kWh',0.0,10.0,0.12,0.01),emission_factor_kg_per_kwh=st.sidebar.number_input('kgCO₂ / kWh',0.0,2.0,0.45,0.01))
with st.sidebar.expander('Maintenance / failure cost assumptions'):
    for _c in COMPONENTS:
        econ.maintenance_cost[_c]=st.number_input(f'{_c}: maintenance cost',0.0,1e7,float(econ.maintenance_cost[_c]),10.0,key=f'mc_{_c}')
        econ.downtime_cost[_c]=st.number_input(f'{_c}: downtime cost',0.0,1e7,float(econ.downtime_cost[_c]),10.0,key=f'dc_{_c}')
        econ.failure_cost[_c]=st.number_input(f'{_c}: failure/replacement risk cost',0.0,1e8,float(econ.failure_cost[_c]),100.0,key=f'fc_{_c}')

st.sidebar.header("5. MOAPO")
opt=OptimizationConfig(population=st.sidebar.number_input('Population',10,100,18,2),iterations=st.sidebar.number_input('Iterations',5,200,10,5),archive_size=st.sidebar.number_input('Pareto archive size',20,300,80,10),horizon_days=st.sidebar.number_input('Prescription horizon (days)',7,180,30,7),decision_interval_days=st.sidebar.number_input('S4 re-optimization interval (days)',1,60,14,1),risk_limit=st.sidebar.slider('Max preferred critical-state risk',0.01,0.50,0.10,0.01))
ctrl=ControlConfig(comfort_ppd_limit=st.sidebar.slider('Comparative PPD limit',5.0,30.0,15.0,1.0))
scfg=StrategyConfig()
with st.sidebar.expander('S0–S3 benchmark policy settings'):
    scfg.s0_threshold=st.slider('S0 reactive threshold',0.70,1.00,0.90,0.01)
    scfg.s2_threshold=st.slider('S2 condition threshold',0.40,0.95,0.70,0.01)
    scfg.s3_base_threshold=st.slider('S3 adaptive base threshold',0.40,0.95,0.70,0.01)
    scfg.s3_forecast_days=st.number_input('S3 projection horizon (days)',1,180,30,1)
    scfg.s3_rul_trigger_days=st.number_input('S3 RUL trigger (days)',1,180,30,1)

st.sidebar.header("6. Experiment")
nmc=st.sidebar.number_input('RUL Monte Carlo runs',100,3000,300,100);rulmax=st.sidebar.number_input('RUL max horizon (days)',180,3650,1825,365);years=st.sidebar.number_input('S0–S4 comparison horizon (years)',0.25,10.0,1.0,0.25)

if not st.button('Run complete S4 Maintenance 4.0 study',type='primary',use_container_width=True):st.stop()
try:
    with st.spinner('Running UKF Digital Twin, Monte Carlo RUL, MOAPO, and S0–S4 comparison...'):
        res=full_analysis(df,cmap,twin,econ,ctrl,opt,scfg,n_rul_mc=int(nmc),rul_max_days=int(rulmax),strategy_years=float(years))
except Exception as e:st.exception(e);st.stop()

st.success('Research run completed.')
ss=res['strategy_summary'].set_index('strategy');s4=ss.loc['S4'];s0=ss.loc['S0'];es=100*(s0['annualized_energy_MWh']-s4['annualized_energy_MWh'])/max(s0['annualized_energy_MWh'],1e-9)
q1,q2,q3,q4,q5=st.columns(5);q1.metric('S4 vs S0 energy',f'{es:.2f}%');q2.metric('S4 annual energy',f"{s4['annualized_energy_MWh']:.1f} MWh/y");q3.metric('S4 actions',int(s4['maintenance_and_replacement_actions']));q4.metric('S4 max final degradation',f"{s4['final_max_degradation']:.3f}");q5.metric('S4 mean PPD proxy',f"{s4['mean_PPD_proxy']:.1f}%")

tabs=st.tabs(['Digital Twin','Probabilistic RUL','S4 MOAPO','S0–S4 comparison','Interventions','Export'])
with tabs[0]:
    st.markdown('### Online degradation-state estimation')
    st.dataframe(res['observability'],use_container_width=True)
    st.warning('Low sensor coverage means the corresponding state is primarily open-loop physics-informed, not measurement-validated.')
    h=res['twin_history'].set_index('timestamp')[[f'{c}_state' for c in COMPONENTS]];st.line_chart(h)
    st.dataframe(res['twin_history'].tail(100),use_container_width=True)
with tabs[1]:
    st.dataframe(res['rul_summary'],use_container_width=True)
    st.bar_chart(res['rul_summary'].set_index('component')['RUL_p50_days'].replace(np.inf,np.nan))
with tabs[2]:
    st.markdown('### Pareto archive')
    st.dataframe(res['pareto'],use_container_width=True)
    st.markdown('### Representative prescriptions')
    st.dataframe(res['representatives'],use_container_width=True)
    st.line_chart(res['moapo_history'].set_index('iteration')[[c for c in ['archive_size','best_energy_cost','best_risk'] if c in res['moapo_history']]])
with tabs[3]:
    st.dataframe(res['strategy_summary'],use_container_width=True)
    st.bar_chart(res['strategy_summary'].set_index('strategy')[['annualized_energy_MWh']])
    st.line_chart(res['strategy_history'].pivot(index='day',columns='strategy',values='max_state'))
with tabs[4]:
    st.dataframe(res['interventions'],use_container_width=True)
    if len(res['interventions']):st.dataframe(pd.crosstab(res['interventions']['strategy'],res['interventions']['component']),use_container_width=True)
with tabs[5]:
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'S4_Maintenance40_Results';write_package(res,out);bio=io.BytesIO();
        with zipfile.ZipFile(bio,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(out.parent))
        payload=bio.getvalue()
    st.download_button('Download complete manuscript results package',payload,'S4_Maintenance40_Results.zip','application/zip',use_container_width=True)
