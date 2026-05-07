import pytest
import pandas as pd
import numpy as np

def test_fraud_rate_reasonable():
    """Fraud rate should be between 1% and 50%"""
    try:
        df = pd.read_csv("data/Fraud_Dataset.csv")
        fraud_rate = df['Is_Fraud'].mean()
        assert 0.01 <= fraud_rate <= 0.50, f"Fraud rate {fraud_rate} is outside expected range"
        print(f"Fraud rate test passed: {fraud_rate:.4f}")
    except FileNotFoundError:
        pytest.skip("Dataset not available in CI environment")

def test_required_columns():
    """Check all required columns exist"""
    try:
        df = pd.read_csv("data/Fraud_Dataset.csv")
        required = ['Transaction_Amount', 'Is_Fraud', 'Credit_Score']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        print("Column test passed!")
    except FileNotFoundError:
        pytest.skip("Dataset not available in CI environment")

def test_no_negative_transaction_amount():
    """Transaction amounts should not be negative"""
    try:
        df = pd.read_csv("data/Fraud_Dataset.csv")
        assert (df['Transaction_Amount'] >= 0).all(), "Negative transaction amounts found!"
        print("Transaction amount test passed!")
    except FileNotFoundError:
        pytest.skip("Dataset not available in CI environment")