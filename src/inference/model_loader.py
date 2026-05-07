import mlflow.pyfunc

def load_production_model():
    return mlflow.pyfunc.load_model("models:/fraud-xgboost/Production")

# For local testing
if __name__ == "__main__":
    model = load_production_model()
    print("Production model loaded successfully")