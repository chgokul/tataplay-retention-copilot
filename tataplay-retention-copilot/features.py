"""Shared feature engineering for training and inference."""
import pandas as pd

SNAPSHOT = pd.Timestamp("2026-03-31")

NUMERIC = [
    "tenure_months", "age", "add_on_count", "monthly_recharge_inr",
    "hours_watched_per_week", "hd_hours_per_week", "sports_hours_per_week",
    "channels_watched", "support_calls_90d", "n_success_6m", "n_failed_6m",
    "recency_days",
]
CATEGORICAL = ["city", "plan_id", "payment_mode", "auto_renew"]
LEAKAGE = ["closure_request_logged", "next_cycle_recharge_inr"]

def build_features(subscribers, recharges):
    """Join six-month recharge aggregates to subscriber rows without using leakage columns."""
    s = subscribers.copy()
    r = recharges.copy()
    r["recharge_date"] = pd.to_datetime(r["recharge_date"])
    r = r[r["recharge_date"] <= SNAPSHOT]
    agg = r.assign(
        success=(r.status == "success").astype(int),
        failed=(r.status == "failed").astype(int),
    ).groupby("subscriber_id").agg(
        n_success_6m=("success", "sum"),
        n_failed_6m=("failed", "sum"),
        last_success=("recharge_date", lambda x: x[r.loc[x.index, "status"].eq("success")].max()
                      if r.loc[x.index, "status"].eq("success").any() else pd.NaT),
    ).reset_index()
    out = s.merge(agg, on="subscriber_id", how="left")
    out[["n_success_6m", "n_failed_6m"]] = out[["n_success_6m", "n_failed_6m"]].fillna(0)
    out["recency_days"] = (SNAPSHOT - pd.to_datetime(out["last_success"])).dt.days
    return out.drop(columns=["last_success"])

def feature_columns():
    """Return the approved numeric and categorical feature names."""
    return NUMERIC, CATEGORICAL
