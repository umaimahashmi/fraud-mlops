import mlflow
import mlflow.xgboost
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix

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

    # Load model
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/xgboost_model")

    # Predictions
    preds_prob = model.predict_proba(X_test)[:, 1]
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
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # SHAP - FIX: extract booster from XGBoost model
    try:
        print("Generating SHAP values...")
        
        # Fix: get the raw booster object
        booster = model.get_booster()
        
        # Use booster directly with SHAP
        explainer = shap.TreeExplainer(booster)
        sample = X_test.iloc[:500]
        shap_values = explainer.shap_values(sample)

        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")
        plt.close()

        print("SHAP completed successfully")

    except Exception as e:
        print("SHAP failed with error:", str(e))
        print("Skipping SHAP - continuing pipeline")

    print("Evaluation Completed")
    print("=" * 60)