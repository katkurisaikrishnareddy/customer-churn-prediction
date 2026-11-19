"""
app.py — Flask web application for Customer Churn Prediction.

Pages     /            single-customer form          /batch     upload a CSV, score many customers
          /metrics     model performance dashboard   /about     methodology
API       POST /api/predict   (one JSON object, or a list)      GET /api/schema   GET /health
"""
import base64
import json
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.predictor import ChurnPredictor, ValidationError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = {
    "high": {"SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
             "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
             "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
             "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 95.7},
    "low": {"SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
            "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
            "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 72.4},
    "medium": {"SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 14, "PhoneService": "Yes", "MultipleLines": "Yes",
               "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
               "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
               "PaperlessBilling": "Yes", "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 84.5},
    "senior": {"SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "tenure": 38, "PhoneService": "Yes", "MultipleLines": "No",
               "InternetService": "No", "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
               "DeviceProtection": "No internet service", "TechSupport": "No internet service", "StreamingTV": "No internet service",
               "StreamingMovies": "No internet service", "Contract": "Month-to-month",
               "PaperlessBilling": "No", "PaymentMethod": "Mailed check", "MonthlyCharges": 20.25},
}


def create_app(artifacts_dir: str = "artifacts") -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024        # 2 MB upload cap
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    predictor = ChurnPredictor(os.path.join(BASE_DIR, artifacts_dir))
    with open(os.path.join(BASE_DIR, artifacts_dir, "evaluation_report.json"), encoding="utf-8") as f:
        report = json.load(f)
    app.predictor, app.report = predictor, report

    @app.route("/")
    def index():
        return render_template("index.html", options=predictor.options, examples=EXAMPLES, form=request.args)

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.form.to_dict()
        try:
            result = predictor.predict_one(data)
            return render_template("result.html", r=result, threshold=predictor.threshold)
        except ValidationError as e:
            return render_template("index.html", options=predictor.options, examples=EXAMPLES, errors=e.errors, form=data), 400

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Send a JSON body (object or list of objects)."}), 400
        records = payload if isinstance(payload, list) else [payload]
        if not records or len(records) > 1000:
            return jsonify({"error": "Provide between 1 and 1000 customer objects."}), 400
        out = []
        try:
            for rec in records:
                if not isinstance(rec, dict):
                    raise ValidationError(["Each customer must be a JSON object"])
                out.append(predictor.predict_one(rec, explain=len(records) == 1))
        except ValidationError as e:
            return jsonify({"error": "Invalid input", "details": e.errors}), 400
        keep = ["churn_probability", "risk_level", "prediction", "label", "threshold", "notes", "drivers_up", "drivers_down", "actions"]
        clean = [{k: r[k] for k in keep if k in r} for r in out]
        return jsonify(clean[0] if not isinstance(payload, list) else clean)

    @app.route("/batch", methods=["GET", "POST"])
    def batch():
        if request.method == "GET":
            return render_template("batch.html")
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith(".csv"):
            return render_template("batch.html", error="Please upload a .csv file."), 400
        try:
            df = pd.read_csv(f)
        except Exception:
            return render_template("batch.html", error="Could not read that CSV file."), 400
        if df.empty or len(df) > 15000:
            return render_template("batch.html", error="CSV must contain between 1 and 15,000 rows."), 400
        scored, bad = predictor.predict_batch(df)
        ok = scored[scored["risk_tier"] != ""]
        summary = {"rows": len(df), "scored": len(ok), "invalid": len(bad),
                   "flagged": int((ok["flag_for_retention"] == "Yes").sum()),
                   "high": int((ok["risk_tier"] == "High").sum()), "medium": int((ok["risk_tier"] == "Medium").sum()),
                   "low": int((ok["risk_tier"] == "Low").sum()),
                   "avg_prob": round(float(ok["churn_probability"].mean() * 100), 1) if len(ok) else 0}
        idcol = "customerID" if "customerID" in scored.columns else None
        cols = ([idcol] if idcol else []) + ["tenure", "Contract", "MonthlyCharges", "churn_probability", "risk_tier"]
        top = ok.sort_values("churn_probability", ascending=False).head(25)[[c for c in cols if c in ok.columns]]
        csv_b64 = base64.b64encode(scored.to_csv(index=False).encode()).decode()
        return render_template("batch.html", summary=summary, top=top.to_dict("records"), cols=list(top.columns), bad=bad[:15], csv_b64=csv_b64)

    @app.route("/sample.csv")
    def sample_csv():
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test.csv")).head(60).drop(columns=["Churn"])
        return app.response_class(df.to_csv(index=False), mimetype="text/csv",
                                  headers={"Content-Disposition": "attachment; filename=sample_customers.csv"})

    @app.route("/metrics")
    def metrics():
        return render_template("metrics.html", r=report, meta=predictor.meta)

    @app.route("/about")
    def about():
        return render_template("about.html", r=report, meta=predictor.meta)

    @app.route("/api/schema")
    def schema():
        return jsonify({**predictor.meta["schema"], "threshold": predictor.threshold, "example": EXAMPLES["high"]})

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "model": report.get("model_name", "unknown"), "threshold": predictor.threshold,
                        "trained_at": predictor.meta.get("trained_at")})

    @app.errorhandler(413)
    def too_large(_):
        return render_template("batch.html", error="File too large (limit 2 MB)."), 413

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() in ("1", "true", "yes")
    print(f"\n" + "=" * 60)
    print(f"  Customer Churn Prediction System")
    print(f"  Live at: http://localhost:{port}")
    print(f"  Metrics: http://localhost:{port}/metrics")
    print(f"  Batch:   http://localhost:{port}/batch")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
