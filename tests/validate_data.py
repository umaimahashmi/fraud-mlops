import os
import sys

def validate_data():
    print("Running data validation checks...")
    
    # Check if data file exists
    if not os.path.exists("data/Fraud_Dataset.csv"):
        print("WARNING: Fraud_Dataset.csv not found - skipping validation")
        return True
    
    import pandas as pd
    df = pd.read_csv("data/Fraud_Dataset.csv")
    
    # Check required columns
    required_columns = [
        'Transaction_ID', 'Customer_ID', 'Transaction_Amount',
        'Transaction_Type', 'Transaction_Location', 'Transaction_Time',
        'Device_Used', 'Account_Age', 'Credit_Score',
        'Previous_Fraud', 'Is_Fraud'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"FAILED: Missing columns: {missing_cols}")
        sys.exit(1)
    
    # Check no empty dataset
    if len(df) == 0:
        print("FAILED: Dataset is empty!")
        sys.exit(1)
    
    # Check target column exists
    if 'Is_Fraud' not in df.columns:
        print("FAILED: Target column Is_Fraud not found!")
        sys.exit(1)
    
    print(f"Data validation passed! Rows: {len(df)}, Columns: {len(df.columns)}")
    return True

if __name__ == "__main__":
    validate_data()