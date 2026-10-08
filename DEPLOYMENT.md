# Deployment

## Windows
1. Extract the ZIP.
2. Install Python 3.11 or 3.12.
3. Double-click `run_local.bat`.

## Command line
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud
Upload the package files to a GitHub repository and select `streamlit_app.py` as the entry point. The included `.streamlit/config.toml` sets a 2048 MB upload limit.

## Quick validation
```bash
python s4_maintenance40_engine.py --demo --output_dir demo_results
```

For publication-scale S4 runs, start with a small optimization budget to verify the pipeline, then increase population/iterations and repeat the MOAPO experiment under multiple random seeds.
