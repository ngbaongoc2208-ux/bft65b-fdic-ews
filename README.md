# BFT65B – A Big-Data Pipeline for Bank Failure Early Warning (FDIC data)

Individual final report. **Author:** Nguyễn Bảo Ngọc (Student ID: `11231002`).  
**Repository:** https://github.com/ngbaongoc2208-ux/bft65b-fdic-ews

The pipeline downloads the public FDIC Call Reports (1984Q1–2026Q2, 1.68 million bank-quarters), the Failures and Institutions tables, stores them as Parquet, audits their quality, builds a clean point-in-time panel with a 4-quarter failure label, benchmarks Polars against pandas and Spark, builds 47 leak-free features, selects a model on a rolling-origin validation, sets an alarm threshold from failure costs and evaluates once on 2012Q1–2025Q2. No data files are committed (they are re-created by the scripts).

## Setup

* Python 3.10 (tested with 3.10.9), Windows 10/11. Pinned libraries are in `requirements.txt`.

    python -m venv .venv
    .venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
    pip install -r requirements.txt

* Run every command from the repository root. Random seeds are fixed (`42`).
* If the repository is inside a synchronised folder (OneDrive), pause synchronisation while scripts 04, 05, 07 and 08 write large files (a clean-panel file was once corrupted during synchronisation).
* PySpark is not needed locally: the Spark parts run in Google Colab (free) with the notebooks below.

## Run Order

Approximate run time on a 12-core laptop with 16 GB RAM:

| Step | Command | Output | Time |
| :--- | :--- | :--- | :--- |
| 1 | `python src/01_download.py` | `data_raw/fdic/` (resumable; use `--start 2023-03-31 --end 2023-12-31` for a quick test) | ~1 h |
| 2 | `python src/02_storage_benchmark.py` | `reports/storage_benchmark.csv` (CSV vs Parquet) | 5–15 min |
| 3 | `python src/03_quality_report.py` | `reports/quality/` (automated audit, figures) | 2–5 min |
| 4 | `python src/04_build_panel.py` | `data_clean/panel.parquet` (clean panel, exits, label) | 3–5 min |
| 5 | `python src/05_polars_vs_pandas.py` | `reports/part3/` (benchmark, query plans, `polars_check.csv`) | 10–20 min |
| 6 | `python src/06_export_for_spark.py` | `data_clean/panel_spark.parquet`; then run `notebooks/03_spark_demo.ipynb` in Colab (upload this file and `reports/part3/polars_check.csv`) | 1 min |
| 7 | `python src/07_features.py` | `data_clean/features.parquet`, `reports/part4/` (47 features, leakage tests) | 12–20 min |
| 8 | `python src/08_model_selection.py` | `reports/part5/frozen_config.json`, validation tables, `data_clean/mllib_sample.parquet`; then run `notebooks/04_mllib_demo.ipynb` in Colab (upload the sample) | 15–30 min |
| 9 | `python src/09_final_test.py` | `reports/part5/test_*.csv`, `test_results.json` | 5–10 min |

## Notes

* `src/fdic_features.py` holds the feature definitions and the training-only preprocessing pipeline (imported by 07, 08 and 09).
* Commit `reports/part5/frozen_config.json` before running step 9. The test period is evaluated once: step 9 checks the hash of the frozen file and refuses to run again while `reports/part5/test_results.json` exists (delete that file only to reproduce the whole pipeline from scratch; `--force` is for a crash only).
* Notebooks in `notebooks/` contain their outputs. Colab sessions print a file size and MD5 to check that the uploaded file is intact.
* The API download is live data: row counts may differ slightly from the report if the FDIC revises its tables (the report uses the download of 14 November 2026).

## Repository Layout

    src/        01_download.py ... 09_final_test.py, fdic_features.py
    notebooks/  03_spark_demo.ipynb (PySpark vs Polars), 04_mllib_demo.ipynb (Spark MLlib vs scikit-learn)
    reports/    benchmark and audit tables, figures, query plans, frozen_config.json, test results (small files)

## Sources and Academic Integrity

* **FDIC BankFind Suite API and field dictionary:** https://api.fdic.gov/banks/docs/ ; **FDIC Bank Failures and Assistance Data:** https://catalog.data.gov/dataset/bank-failures-and-assistance-data
* `src/01_download.py` is the course starter script `fdic_download_starter.py`, adapted by the author.
* **Libraries:** Polars, pandas, NumPy, PyArrow, scikit-learn, matplotlib, psutil, Apache Spark / PySpark (Colab).
* **AI assistance:** Large language models (Claude, Gemini) were consulted solely as technical reference tools for code optimization and English language refinement. All methodological choices, data engineering pipelines, empirical experiments, and final conclusions were independently conceived, executed, and verified by the author.