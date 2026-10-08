"""
train_models.py
Trains the ML method that powers the agent's decision:
  - naive benchmark: median price of same make/model/model year (previous 12 months)
  - ridge regression on log price
  - gradient boosting (quantile 10%, 50%, 90%) on log price -> price range
Time-based split: train before July 2025, test July 2025 - June 2026.
Run:  python train_models.py
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

CUTOFF = pd.Timestamp("2025-07-01")
CAT = ["make", "model", "fuel"]
NUM = ["age", "odometer", "miles_per_year", "original_price", "orig_missing", "month_index"]


def add_features(df):
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["age"] = (df["sale_date"].dt.year - df["model_year"]).clip(lower=0)
    df["miles_per_year"] = df["odometer"] / df["age"].clip(lower=1)
    df["orig_missing"] = df["original_price"].isna().astype(int)
    df["month_index"] = (df["sale_date"].dt.year - 2019) * 12 + df["sale_date"].dt.month
    return df


def metrics(actual, pred):
    ape = np.abs(pred - actual) / actual
    return {"MdAPE": round(float(np.median(ape)) * 100, 1),
            "within_10%": round(float(np.mean(ape <= 0.10)) * 100, 1),
            "within_20%": round(float(np.mean(ape <= 0.20)) * 100, 1)}


if __name__ == "__main__":
    df = add_features(pd.read_csv("used_resales_clean.csv"))
    train, test = df[df.sale_date < CUTOFF], df[df.sale_date >= CUTOFF]
    print(f"Train: {len(train):,} resales | Test: {len(test):,} resales")
    y_train = np.log(train["price"])

    # ---- naive benchmark
    recent = train[train.sale_date >= CUTOFF - pd.DateOffset(months=12)]
    bench = recent.groupby(["make", "model", "model_year"])["price"].median()
    b = test.join(bench.rename("bench"), on=["make", "model", "model_year"])
    b = b.dropna(subset=["bench"])
    print("Naive benchmark   ", metrics(b["price"], b["bench"]), f"(covers {len(b)/len(test):.0%} of test)")

    # ---- ridge regression
    prep_ridge = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM)])
    ridge = make_pipeline(prep_ridge, Ridge(alpha=1.0))
    ridge.fit(train[CAT + NUM], y_train)
    print("Ridge regression  ", metrics(test["price"], np.exp(ridge.predict(test[CAT + NUM]))))

    # ---- gradient boosting, quantile models
    def gb(q):
        prep = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
            ("num", "passthrough", NUM)])
        model = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                              learning_rate=0.05, categorical_features=[0, 1, 2],
                                              random_state=28)
        return make_pipeline(prep, model).fit(train[CAT + NUM], y_train)

    gb_low, gb_mid, gb_high = gb(0.10), gb(0.50), gb(0.90)
    lo = np.exp(gb_low.predict(test[CAT + NUM]))
    mid = np.exp(gb_mid.predict(test[CAT + NUM]))
    hi = np.exp(gb_high.predict(test[CAT + NUM]))
    print("Gradient boosting ", metrics(test["price"], mid))
    print(f"Range coverage (10th-90th): {np.mean((test.price >= lo) & (test.price <= hi)):.0%} (target ~80%)")

    joblib.dump({"ridge": ridge, "low": gb_low, "mid": gb_mid, "high": gb_high,
                 "features": CAT + NUM}, "price_models.pkl")
    print("Saved price_models.pkl")
