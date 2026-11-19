.PHONY: install train train-fast test check run docker notebook
install:        ; pip install -r requirements-train.txt
train:          ; python run_pipeline.py
train-fast:     ; python run_pipeline.py --fast
test:           ; python -m pytest -q
check:          ; python -c "import flask, joblib, numpy, pandas, sklearn; print('Dependencies OK')" && python -m pytest -q
run:            ; python app.py
docker:         ; docker build -t churn-prediction . && docker run -p 5000:5000 churn-prediction
notebook:       ; jupyter nbconvert --to notebook --execute --inplace notebooks/01_end_to_end_churn_analysis.ipynb
