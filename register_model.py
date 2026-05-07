import mlflow
from mlflow.tracking import MlflowClient

def register_best_model():
    client = MlflowClient()
    
    # Search for best run
    runs = mlflow.search_runs(
        order_by=["metrics.recall_fraud DESC", "start_time DESC"],
        max_results=5
    )
    
    if runs.empty:
        print("No runs found!")
        return
    
    best_run = runs.iloc[0]
    best_run_id = best_run['run_id']
    
    recall = best_run.get("metrics.recall_fraud", 0.0)
    auc = best_run.get("metrics.auc", 0.0)
    
    print("Best Run ID     :", best_run_id)
    print("Recall (Fraud)  :", round(recall, 4))
    print("AUC             :", round(auc, 4))
    
    # Register model
    model_uri = f"runs:/{best_run_id}/xgboost_model"
    
    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri, 
            name="fraud-xgboost"
        )
        
        # Move to Production
        client.transition_model_version_stage(
            name="fraud-xgboost",
            version=registered_model.version,
            stage="Production"
        )
        
        print("Model successfully registered in PRODUCTION stage!")
        print("Version:", registered_model.version)
        
    except Exception as e:
        print("Registration failed:", str(e))

if __name__ == "__main__":
    register_best_model()