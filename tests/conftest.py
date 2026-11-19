import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, pytest
from src.utils import load_config, abs_path

@pytest.fixture(scope="session")
def config():
    return load_config()

@pytest.fixture(scope="session")
def raw_df(config):
    return pd.read_csv(abs_path(config["data"]["raw_path"]))

HIGH = {"SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.7}
LOW = {**HIGH, "Partner": "Yes", "Dependents": "Yes", "tenure": 60, "InternetService": "DSL", "OnlineSecurity": "Yes",
       "TechSupport": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
       "Contract": "Two year", "PaymentMethod": "Credit card (automatic)", "PaperlessBilling": "No", "MonthlyCharges": 72.4}

@pytest.fixture
def high(): return dict(HIGH)

@pytest.fixture
def low(): return dict(LOW)
