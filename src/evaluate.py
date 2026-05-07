import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix

with mlflow.start_run(nested=True, run_name="model_evaluation"):
    X_test = pd.read_parquet("data/X_test_final.parquet")
    y_test = pd.read_parquet("data/y_test.parquet").iloc[:, 0]
    
    # === FIXED: Search for the latest training run ===
    runs = mlflow.search_runs(
        order_by=["start_time DESC"],
        max_results=5
    )
    
    # Find the most recent training run
    train_run = runs[runs["tags.mlflow.runName"] == "xgboost_training"]
    if train_run.empty:
        train_run = runs.iloc[0]  # fallback to latest run
    
    run_id = train_run.iloc[0].run_id
    
    print("Loading model from run ID:", run_id)
    
    # Load model
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/xgboost_model")
    
    # Predictions
    preds_prob = model.predict_proba(X_test)[:, 1]
    preds = (preds_prob > 0.5).astype(int)
    
    # Metrics
    metrics = {
        "auc": roc_auc_score(y_test, preds_prob),
        "recall_fraud": recall_score(y_test, preds),
        "precision": precision_score(y_test, preds),
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
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # SHAP (using sample for speed)
    print("Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:800])
    
    shap.summary_plot(shap_values, X_test.iloc[:800], show=False)
    plt.savefig("shap_summary.png")
    mlflow.log_artifact("shap_summary.png")
    
    print("Evaluation Completed with SHAP")
    print("="*50)