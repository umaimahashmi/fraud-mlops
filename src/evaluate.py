import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, recall_score,
                             precision_score, f1_score,
                             confusion_matrix)

with mlflow.start_run(nested=True, run_name="model_evaluation"):
    X_test = pd.read_parquet("data/X_test_final.parquet")
    y_test = pd.read_parquet("data/y_test.parquet").iloc[:, 0]

    # Find latest training run
    runs = mlflow.search_runs(
        order_by=["start_time DESC"],
        max_results=10
    )

    train_runs = runs[runs["tags.mlflow.runName"] == "xgboost_training"]
    if not train_runs.empty:
        run_id = train_runs.iloc[0]["run_id"]
    else:
        run_id = runs.iloc[0]["run_id"]

    print("Loading model from run ID:", run_id)

    # Load as booster directly to avoid version issues
    import xgboost as xgb
    import mlflow.artifacts
    import os
    import tempfile

    # Load model using xgboost booster directly
    model_uri = f"runs:/{run_id}/xgboost_model"
    
    try:
        # Try loading as pyfunc first
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        
        # Get predictions using booster
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        booster = xgb.Booster()
        booster.load_model(os.path.join(local_path, "model.xgb"))
        
        dtest = xgb.DMatrix(X_test)
        preds_prob = booster.predict(dtest)
        
    except Exception as e:
        print(f"Booster load failed: {e}, trying sklearn interface...")
        # Fallback: load sklearn model
        model = mlflow.xgboost.load_model(model_uri)
        # Use decision function instead of predict_proba
        dtest = xgb.DMatrix(X_test)
        booster = model.get_booster()
        preds_prob = booster.predict(dtest)

    preds = (preds_prob > 0.4).astype(int)

    # Metrics
    metrics = {
        "auc": roc_auc_score(y_test, preds_prob),
        "recall_fraud": recall_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds)
    }

    mlflow.log_metrics(metrics)

    print("Model Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k:15}: {v:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Fraud Detection')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # SHAP
    try:
        import shap
        print("Generating SHAP values...")
        explainer = shap.TreeExplainer(booster)
        sample = X_test.iloc[:300]
        dmatrix_sample = xgb.DMatrix(sample)
        shap_values = explainer.shap_values(sample)

        plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")
        plt.close()
        print("SHAP completed successfully")

    except Exception as e:
        print(f"SHAP skipped: {e}")

    print("Evaluation Completed")
    print("=" * 60)