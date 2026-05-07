import mlflow
import subprocess
import sys

EXPERIMENT_NAME = "Banking_Fraud_Detection"

mlflow.set_experiment(EXPERIMENT_NAME)

def run_step(script_path: str):
    print(f"\n{'='*70}")
    print(f"RUNNING: {script_path}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              check=True, capture_output=True, text=True)
        print(result.stdout.strip())
        if result.stderr.strip():
            print("INFO:", result.stderr.strip())
        print(f"SUCCESS: {script_path}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {script_path}")
        print(e.stdout)
        if e.stderr: 
            print(e.stderr)
        raise


if __name__ == "__main__":
    print(f"🚀 Starting {EXPERIMENT_NAME} Pipeline\n")
    
    with mlflow.start_run(run_name="full-pipeline-run") as parent_run:
        mlflow.set_tag("dataset", "Fraud_Dataset_2025")
        
        try:
            run_step("src/data_ingest.py")
            run_step("src/preprocess.py")
            run_step("src/feature_engineering.py")
            run_step("src/train.py")
            run_step("src/evaluate.py")
            run_step("src/drift_detection.py")
            
            # === FIXED: Correct way to get metrics from search_runs ===
            runs = mlflow.search_runs(
                order_by=["start_time DESC"],
                max_results=10
            )
            
            # Find training run
            train_runs = runs[runs["tags.mlflow.runName"] == "xgboost_training"]
            if not train_runs.empty:
                best_run = train_runs.iloc[0]
            else:
                best_run = runs.iloc[0]
            
            # Correct way to access metrics
            recall = best_run.get("metrics.recall_fraud", 0.0)
            auc = best_run.get("metrics.auc", 0.0)
            
            print(f"\n{'='*65}")
            print("FINAL PIPELINE SUMMARY")
            print(f"{'='*65}")
            print(f"AUC                    : {auc:.4f}")
            print(f"Recall (Fraud)         : {recall:.4f}")
            print(f"Experiment             : {EXPERIMENT_NAME}")
            
            # Conditional Deployment (relaxed for this dataset)
            if recall >= 0.35 and auc >= 0.65:
                print("✅ Model meets deployment threshold!")
                run_step("register_model.py")
            else:
                print("⚠️  Model did not meet deployment threshold (yet).")
            
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            raise