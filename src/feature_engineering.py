import pandas as pd
import mlflow

with mlflow.start_run(nested=True, run_name="feature_engineering"):
    X_train = pd.read_parquet("data/X_train.parquet")
    y_train = pd.read_parquet("data/y_train.parquet").iloc[:, 0]
    X_test = pd.read_parquet("data/X_test.parquet")
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # === FIX: Handle high cardinality better ===
    # Drop Transaction_Location (too noisy/high cardinality)
    if 'Transaction_Location' in X_train.columns:
        X_train = X_train.drop('Transaction_Location', axis=1)
        X_test = X_test.drop('Transaction_Location', axis=1)
        print("Dropped Transaction_Location (high cardinality)")
    
    # One-hot encoding for categorical columns
    cat_cols = ['Transaction_Type', 'Device_Used']
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    
    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    X_train.to_parquet("data/X_train_final.parquet")
    X_test.to_parquet("data/X_test_final.parquet")
    
    print("Feature Engineering Completed")
    print("Final Features Count:", X_train.shape[1])
    print("Final Columns:", X_train.columns.tolist())
    print("="*60)