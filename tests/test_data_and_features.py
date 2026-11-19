import numpy as np, pandas as pd, pytest
from src.data_validation import DataValidation
from src.feature_engineering import ChurnFeatureEngineer, clean_total_charges, get_feature_lists
from src.data_preprocessing import DataPreprocessing, build_preprocessor


def test_raw_data_shape(raw_df):
    assert raw_df.shape == (7043, 21) and raw_df.customerID.is_unique


def test_validation_passes_on_real_data(config, raw_df):
    assert DataValidation(config).run(raw_df) is True


def test_validation_rejects_bad_category(config, raw_df):
    bad = raw_df.copy(); bad.loc[0, "Contract"] = "Weekly"
    with pytest.raises(ValueError):
        DataValidation(config).run(bad)


def test_validation_rejects_missing_column(config, raw_df):
    with pytest.raises(ValueError):
        DataValidation(config).run(raw_df.drop(columns=["tenure"]))


def test_total_charges_blank_new_customers_are_zero(raw_df):
    tc = clean_total_charges(raw_df)
    assert tc.isna().sum() == 0
    assert (tc[raw_df.tenure == 0] == 0).all()          # v1 imputed the median here (a bug)


def test_feature_engineering_columns_and_no_nulls(raw_df):
    out = ChurnFeatureEngineer(True, ["gender"]).transform(raw_df)
    assert "gender" not in out.columns and "customerID" not in out.columns and "Churn" not in out.columns
    assert out.isna().sum().sum() == 0
    num, cat = get_feature_lists(True, ["gender"])
    assert set(num + cat) <= set(out.columns)
    assert out["num_addon_services"].between(0, 6).all()


def test_feature_engineering_is_stateless(raw_df):
    fe = ChurnFeatureEngineer(True, None)
    pd.testing.assert_frame_equal(fe.fit(raw_df.head(100)).transform(raw_df.tail(50)), fe.transform(raw_df.tail(50)))


def test_split_is_stratified_and_leak_free(config, raw_df):
    Xtr, Xte, ytr, yte = DataPreprocessing(config).split_data(raw_df)
    assert len(Xtr) + len(Xte) == len(raw_df)
    assert set(Xtr.customerID).isdisjoint(set(Xte.customerID))
    assert abs(ytr.mean() - yte.mean()) < 0.005


def test_scaler_fitted_on_training_only(config, raw_df):
    """Regression test for the v1 leakage bug: scaler statistics must come from the training rows only."""
    Xtr, Xte, ytr, _ = DataPreprocessing(config).split_data(raw_df)
    pre = build_preprocessor(config)
    fe = ChurnFeatureEngineer(True, ["gender"])
    pre.fit(fe.transform(Xtr))
    scaler = pre.named_transformers_["num"]
    num, _ = get_feature_lists(True, ["gender"])
    expected = fe.transform(Xtr)[num].mean().values
    np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-6)
