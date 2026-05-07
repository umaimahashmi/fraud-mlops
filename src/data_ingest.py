import pandas as pd
import mlflow

with mlflow.start_run(nested=True, run_name="data_ingestion"):
    # Correct filename
    df = pd.read_csv("data/Fraud_Dataset.csv")
    
    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Fraud Rate:", round(df["Is_Fraud"].mean(), 4))
    
    # Feature extraction from datetime
    df['Transaction_Time'] = pd.to_datetime(df['Transaction_Time'])
    df['Transaction_Hour'] = df['Transaction_Time'].dt.hour
    df['Transaction_DayOfWeek'] = df['Transaction_Time'].dt.dayofweek
    
    df.to_parquet("data/merged_data.parquet", index=False)
    
    mlflow.log_metric("rows", len(df))
    mlflow.log_metric("fraud_rate", df["Is_Fraud"].mean())
    mlflow.log_metric("columns", len(df.columns))
    
    print("Data Ingestion Completed Successfully")
    print("="*50)