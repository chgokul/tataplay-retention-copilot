"""Part C: subscriber clustering, spend modelling and revenue-at-risk ranking."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"

def run():
    """Fit segments, a next-cycle spend model and ranked risk list."""
    OUT.mkdir(exist_ok=True)
    s=pd.read_csv(ROOT/"data/clean/subscribers.csv")
    cols=["tenure_months","hours_watched_per_week","monthly_recharge_inr"]
    scaler=StandardScaler(); z=scaler.fit_transform(s[cols])
    rows=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(z)
        rows.append({"k":k,"inertia":km.inertia_,"silhouette":silhouette_score(z,km.labels_)})
    scores=pd.DataFrame(rows); print("K SCORES\n",scores.to_string(index=False))
    k=4
    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(z)
    s["segment_id"]=km.labels_
    profile=s.groupby("segment_id").agg(size=("subscriber_id","size"),
      mean_tenure=("tenure_months","mean"),mean_hours=("hours_watched_per_week","mean"),
      mean_monthly=("monthly_recharge_inr","mean"),churn_rate=("churned","mean")).reset_index()
    # Assign names from measurable profiles; cluster IDs are arbitrary labels.
    risk_id=int(profile.loc[profile.churn_rate.idxmax(),"segment_id"])
    premium_id=int(profile.loc[profile.mean_monthly.idxmax(),"segment_id"])
    names={}
    for _,row in profile.iterrows():
        sid=int(row.segment_id)
        names[sid]="At-Risk" if sid==risk_id else ("Premium" if sid==premium_id else ("Engaged" if row.mean_hours>18 else "Value"))
    s["segment_name"]=s.segment_id.map(names)
    s[["subscriber_id","segment_id","segment_name"]].to_csv(OUT/"segments.csv",index=False)
    print("SEGMENT PROFILE\n",profile.to_string(index=False))
    new=scaler.transform([[4,15,520]])[0]
    distances=np.linalg.norm(km.cluster_centers_-new,axis=1)
    print("NEW SUBSCRIBER",int(np.argmin(distances)),distances.tolist(),"predict",int(km.predict([new])[0]))
    # Spend model: only non-churned rows have next-cycle recharge labels.
    labelled=s[pd.to_numeric(s.next_cycle_recharge_inr,errors="coerce").notna()].copy()
    X=labelled[cols+["hd_hours_per_week","sports_hours_per_week","add_on_count"]]
    y=pd.to_numeric(labelled.next_cycle_recharge_inr)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    for model in [Ridge(alpha=1.0),Lasso(alpha=1.0,max_iter=10000)]:
        model.fit(Xtr,ytr); pred=model.predict(Xte)
        print(type(model).__name__,{"mae":mean_absolute_error(yte,pred),
          "rmse":mean_squared_error(yte,pred)**.5,"r2":r2_score(yte,pred)})
    # Revenue at risk uses test-set churn probabilities supplied by Part B.
    risk=pd.read_csv(OUT/"test_scores.csv").merge(s[["subscriber_id","monthly_recharge_inr","segment_name"]],on="subscriber_id")
    risk["expected_revenue_at_risk"]=risk.churn_probability*risk.monthly_recharge_inr
    risk=risk.sort_values("expected_revenue_at_risk",ascending=False).head(15)
    risk.to_csv(OUT/"retention_priority.csv",index=False)
    print("TOP REVENUE AT RISK\n",risk.to_string(index=False))

if __name__=="__main__": run()
