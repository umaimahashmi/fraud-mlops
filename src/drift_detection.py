import mlflow
import pandas as pd

with mlflow.start_run(nested=True, run_name="drift_detection"):
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        
        reference = pd.read_parquet("data/X_train_final.parquet")
        current = pd.read_parquet("data/X_test_final.parquet")
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        
        drift_score = report.as_dict()['metrics'][0]['result'].get('dataset_drift_score', 0)
        
        mlflow.log_metric("drift_score", drift_score)
        report.save_html("drift_report.html")
        mlflow.log_artifact("drift_report.html")
        
        print("Data Drift Detection Completed")
        print("Drift Score:", round(drift_score, 4))
        
    except ImportError:
        print("Evidently not installed. Skipping advanced drift report.")
        mlflow.log_metric("drift_score", 0.0)
        print("Basic drift check skipped")
    except Exception as e:
        print("Drift detection failed:", str(e))
        mlflow.log_metric("drift_score", 0.0)
    
    print("="*50)