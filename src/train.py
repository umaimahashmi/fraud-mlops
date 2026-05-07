import os
import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

with mlflow.start_run(run_name="xgboost_training"):
    X_train = pd.read_parquet("data/X_train_final.parquet")
    y_train = pd.read_parquet("data/y_train.parquet").iloc[:, 0]
    X_test = pd.read_parquet("data/X_test_final.parquet")
    y_test = pd.read_parquet("data/y_test.parquet").iloc[:, 0]

    print("Training Fraud Rate:", round(y_train.mean(), 4))
    print("Test Fraud Rate    :", round(y_test.mean(), 4))
    print("Number of Features :", X_train.shape[1])

    model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.2,
        max_depth=10,
        scale_pos_weight=2.5,
        eval_metric='aucpr',
        random_state=42,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0
    )

    model.fit(X_train, y_train)

    preds_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.25

    preds = (preds_prob > threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, preds_prob),
        "recall_fraud": recall_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds),
        "threshold_used": threshold,
        "avg_fraud_prob": float(preds_prob.mean())
    }

    mlflow.log_metrics(metrics)
    mlflow.xgboost.log_model(model, "xgboost_model")

    # Fix: create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model.get_booster().save_model("models/model.xgb")
    mlflow.log_artifact("models/model.xgb")

    # Feature Importance
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\nTop 10 Important Features:")
    print(importance.head(10))

    print("\n=== Training Results ===")
    print("AUC                    :", round(metrics["auc"], 4))
    print("Recall (Fraud)         :", round(metrics["recall_fraud"], 4))
    print("Avg Fraud Probability  :", round(metrics["avg_fraud_prob"], 5))
    print("="*70)