# src/drift_check.py
import json
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, CatTargetDriftPreset, NumTargetDriftPreset
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import sys
import os
import requests

BASELINE_PATH = "data/baseline_stats.json"
BASELINE_DATA = "data/train_sample_for_evidently.csv"  # keep baseline sample for evidently
PROD_SAMPLE = "data/prod_sample.csv"  # a sample of recent production features
DRIFT_OUTPUT = "reports/drift_report.html"

def run_drift_check(baseline_csv=BASELINE_DATA, prod_csv=PROD_SAMPLE):
    if not os.path.exists(baseline_csv) or not os.path.exists(prod_csv):
        print("Baseline or prod sample missing.")
        return False

    ref = pd.read_csv(baseline_csv)
    cur = pd.read_csv(prod_csv)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    os.makedirs(os.path.dirname(DRIFT_OUTPUT), exist_ok=True)
    report.save_html(DRIFT_OUTPUT)
    print(f"Drift report saved to {DRIFT_OUTPUT}")
    # decide drift: you can parse the metrics or set a simple rule
    # For example: if any column drift_score > 0.5 -> drift
    # Use the report JSON if needed
    return True

if __name__ == "__main__":
    ok = run_drift_check()
    if ok:
        print("Drift check completed.")
        # optionally: if drift found -> call API to trigger retrain (see CI section)
