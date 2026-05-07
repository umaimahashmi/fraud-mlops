from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import uvicorn

app = FastAPI(title="Banking Fraud Detection API")

# Load the production model
model = mlflow.pyfunc.load_model("models:/fraud-xgboost/Production")

# Define input schema matching training columns
class Transaction(BaseModel):
    Transaction_Amount: float
    Account_Age: int
    Credit_Score: int
    Previous_Fraud: int
    Transaction_Hour: int
    Transaction_DayOfWeek: int
    Transaction_Type_Online_Payment: int = 0
    Transaction_Type_Transfer: int = 0
    Transaction_Type_Withdrawal: int = 0
    Device_Used_Mobile: int = 0
    Device_Used_POS_Terminal: int = 0
    Device_Used_Web: int = 0


@app.post("/predict")
def predict(data: Transaction):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Make prediction
        prob = float(model.predict(input_df)[0])
        
        return {
            "fraud_probability": round(prob, 4),
            "is_fraud": bool(prob > 0.4),
            "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low",
            "confidence": round(abs(prob - 0.5), 4)
        }
    
    except Exception as e:
        return {"error": str(e), "message": "Prediction failed. Check input data."}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)