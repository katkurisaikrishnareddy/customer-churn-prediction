import pytest
from src.predictor import ChurnPredictor, ValidationError
from app import create_app


@pytest.fixture(scope="module")
def predictor():
    return ChurnPredictor()


@pytest.fixture(scope="module")
def client():
    app = create_app(); app.config["TESTING"] = True
    return app.test_client()


def test_high_risk_beats_low_risk(predictor, high, low):
    ph, pl = predictor.predict_one(high), predictor.predict_one(low)
    assert ph["probability"] > pl["probability"] + 0.4
    assert ph["tier"] == "High" and pl["tier"] == "Low"


def test_probability_in_range_and_deterministic(predictor, high):
    a, b = predictor.predict_one(high), predictor.predict_one(high)
    assert 0 <= a["probability"] <= 1 and a["probability"] == b["probability"] and a["drivers_up"] == b["drivers_up"]


@pytest.mark.parametrize("field,value", [("Contract", "Weekly"), ("tenure", -1), ("tenure", "abc"), ("MonthlyCharges", 9999), ("InternetService", "5G")])
def test_invalid_input_rejected(predictor, high, field, value):
    high[field] = value
    with pytest.raises(ValidationError):
        predictor.predict_one(high)


def test_missing_field_rejected(predictor, high):
    del high["Contract"]
    with pytest.raises(ValidationError):
        predictor.predict_one(high)


def test_internet_consistency_normalised(predictor, high):
    high["InternetService"] = "No"; high["TechSupport"] = "Yes"       # contradictory input
    r = predictor.predict_one(high)
    assert r["input"]["TechSupport"] == "No internet service" and r["notes"]


def test_total_charges_estimated_when_missing(predictor, high):
    r = predictor.predict_one(high)
    assert abs(r["input"]["TotalCharges"] - 3 * 95.7) < 0.01


def test_batch_scoring_flags_bad_rows(predictor, high, low):
    import pandas as pd
    bad = {**high, "Contract": "Weekly"}
    scored, errors = predictor.predict_batch(pd.DataFrame([high, low, bad]))
    assert len(errors) == 1 and errors[0]["row"] == 4
    assert scored.loc[0, "churn_probability"] > scored.loc[1, "churn_probability"]


def test_health_and_pages(client):
    assert client.get("/health").get_json()["status"] == "ok"
    for url in ("/", "/batch", "/metrics", "/about", "/api/schema", "/sample.csv"):
        assert client.get(url).status_code == 200


def test_api_predict_ok_and_errors(client, high):
    r = client.post("/api/predict", json=high); j = r.get_json()
    assert r.status_code == 200 and 0 <= j["churn_probability"] <= 100 and j["actions"]
    assert client.post("/api/predict", json={**high, "Contract": "x"}).status_code == 400
    assert client.post("/api/predict", data="not json", content_type="text/plain").status_code == 400
    assert len(client.post("/api/predict", json=[high, high]).get_json()) == 2


def test_form_predict_page(client, high):
    r = client.post("/predict", data=high)
    assert r.status_code == 200 and b"churn risk assessment" in r.data.lower()
    assert client.post("/predict", data={**high, "tenure": "-3"}).status_code == 400


def test_batch_upload_rejects_non_csv(client):
    import io
    r = client.post("/batch", data={"file": (io.BytesIO(b"x"), "a.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400
