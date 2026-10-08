# START HERE — S4 Maintenance 4.0 Research Software v1

## Fastest way to run

### Windows
Double-click `run_local.bat`.

### Terminal
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### First verification
Choose **Built-in research demonstration** in the app and run with the default settings.

## Then use your research data

1. Choose **Upload CSV/XLSX**.
2. Map the available columns.
3. Confirm which health channels are actually measured.
4. Run the full study.
5. Download the complete results ZIP from the Export tab.

## Publication sequence

Do not start by increasing MOAPO population and iterations. First verify:

1. sensor mapping,
2. UKF state behavior,
3. measurement innovation/residuals,
4. RUL plausibility,
5. Pareto prescription plausibility,
6. S0–S4 intervention counts,
7. energy/comfort/health trade-offs.

After that, increase the optimizer budget and run multiple seeds for publication statistics.
