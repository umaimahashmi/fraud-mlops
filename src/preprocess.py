import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

with mlflow.start_run(nested=True, run_name="preprocessing"):
    df = pd.read_parquet("data/merged_data.parquet")
    
    target_col = "Is_Fraud"
    
    # Drop unnecessary columns
    drop_cols = ['Transaction_ID', 'Customer_ID', 'Transaction_Time']
    X = df.drop([target_col] + drop_cols, axis=1, errors='ignore')
    y = df[target_col]
    
    # Fill missing values
    X = X.fillna(-999)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Save files
    X_train.to_parquet("data/X_train.parquet")
    X_test.to_parquet("data/X_test.parquet")
    y_train.to_frame(name=target_col).to_parquet("data/y_train.parquet")
    y_test.to_frame(name=target_col).to_parquet("data/y_test.parquet")
    
    mlflow.log_metric("train_rows", len(X_train))
    mlflow.log_metric("test_rows", len(X_test))
    mlflow.log_metric("fraud_rate_train", y_train.mean())
    
    print("Preprocessing Completed")
    print("="*50)