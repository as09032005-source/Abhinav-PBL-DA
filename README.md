# TerraLuxe Tiles: Data Analytics Project

Automated tile factory exporting worldwide, with a 7D product viewer and an AI Room Visualizer.

## Run order
```
pip install -r requirements.txt
python 01_generate_synthetic_data.py   # Step 1: creates data/raw/*.csv (with deliberate errors)
python 02_clean_transform.py           # Step 2: creates data/clean/*.csv + data/cleaning_log.csv
python 03_eda_diagnostics.py           # Step 3: creates charts/*.png + outputs/kpis_and_tests.json
streamlit run app.py                   # Interactive dashboard (8 pages)
```

## Folder
- data/raw/        orders_raw.csv, production_raw.csv, sessions_raw.csv (dirty)
- data/clean/      orders_clean.csv, production_clean.csv, sessions_clean.csv
- data/cleaning_log.csv   every cleaning action, rows affected, rationale
- charts/          18 figures used in the report
- outputs/kpis_and_tests.json   all KPIs and hypothesis-test results
