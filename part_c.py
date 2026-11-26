"""Part C: K-means segmentation, regularised spend modelling and revenue at risk."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
SEG_FEATURES=["tenure_months","hours_watched_per_week","monthly_recharge_inr"]
SPEND_FEATURES=["monthly_recharge_inr","tenure_months","hours_watched_per_week","hd_hours_per_week","sports_hours_per_week","channels_watched","support_calls_90d"]

def run():
    """Run the full Part C analysis and save segment and priority CSVs."""
    s=pd.read_csv(ROOT/"data/clean/subscribers.csv")
    scaler=StandardScaler(); z=scaler.fit_transform(s[SEG_FEATURES])
    ks=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(z)
        ks.append({"k":k,"inertia":float(km.inertia_),"silhouette":float(silhouette_score(z,km.labels_))})
    chosen_k=4
    km=KMeans(n_clusters=chosen_k,n_init=10,random_state=42).fit(z)
    s["segment_id"]=km.labels_
    profile=s.groupby("segment_id").agg(size=("subscriber_id","size"),mean_tenure=("tenure_months","mean"),
       mean_hours=("hours_watched_per_week","mean"),mean_monthly=("monthly_recharge_inr","mean"),churn_rate=("churned","mean")).reset_index()
    risk_id=int(profile.loc[profile.churn_rate.idxmax(),"segment_id"])
    premium_id=int(profile.loc[profile.mean_monthly.idxmax(),"segment_id"])
    # Assign descriptive names to remaining clusters based on usage intensity.
    remaining=[int(i) for i in profile.segment_id if int(i) not in (risk_id,premium_id)]
    names={risk_id:"At-Risk",premium_id:"Premium"}
    for sid in remaining:
        row=profile.loc[profile.segment_id==sid].iloc[0]
        names[sid]="Engaged" if row.mean_hours>=profile.mean_hours.median() else "Value"
    s["segment_name"]=s.segment_id.map(names)
    s[["subscriber_id","segment_id","segment_name"]].to_csv(OUT/"segments.csv",index=False)
    profile["segment_name"]=profile.segment_id.map(names)
    profile["recommended_action"]=profile.segment_name.map({"At-Risk":"Proactive retention outreach and recharge-friction support",
       "Premium":"Protect premium experience; offer relevant add-ons","Engaged":"Promote content bundles aligned with viewing","Value":"Offer low-cost plan optimisation"})

    # Hand assignment for the requested example, including every centroid distance.
    new_scaled=scaler.transform(pd.DataFrame([[4,15,520]],columns=SEG_FEATURES))[0]
    distances=np.linalg.norm(km.cluster_centers_-new_scaled,axis=1)
    nearest=int(np.argmin(distances)); predicted=int(km.predict([new_scaled])[0])
    assert nearest==predicted

    # Correlation matrix and pairs over the required threshold.
    corr_cols=["hours_watched_per_week","hd_hours_per_week","sports_hours_per_week","channels_watched"]
    corr=s[corr_cols].corr()
    pairs=[]
    for i,a in enumerate(corr_cols):
        for b in corr_cols[i+1:]:
            if abs(corr.loc[a,b])>.9: pairs.append({"feature_1":a,"feature_2":b,"correlation":float(corr.loc[a,b])})

    # Spend model uses only retained subscribers, with the specified seven predictors.
    keep=s[s.churned==0].copy()
    X=keep[SPEND_FEATURES]; y=pd.to_numeric(keep.next_cycle_recharge_inr)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    alphas=[.001,.01,.1,.5,1,2,3,10,30,100]
    spend_models={}; spend_metrics={}
    for name,estimator in [("linear",LinearRegression()),("ridge",Ridge()),("lasso",Lasso(max_iter=10000))]:
        if name=="linear":
            pipe=Pipeline([("scale",StandardScaler()),("model",estimator)])
            cv_r2=float(cross_val_score(pipe,X_train,y_train,cv=5,scoring="r2").mean())
            model=pipe.fit(X_train,y_train)
            best_alpha=None
        else:
            pipe=Pipeline([("scale",StandardScaler()),("model",estimator)])
            gs=GridSearchCV(pipe,{"model__alpha":alphas},cv=5,scoring="r2",n_jobs=-1)
            gs.fit(X_train,y_train); model=gs.best_estimator_; best_alpha=float(gs.best_params_["model__alpha"]); cv_r2=float(gs.best_score_)
        pred=model.predict(X_test)
        spend_models[name]=model
        spend_metrics[name]={"best_alpha":best_alpha,"cv_r2":cv_r2,"test_r2":float(r2_score(y_test,pred)),
           "mae":float(mean_absolute_error(y_test,pred)),"rmse":float(mean_squared_error(y_test,pred)**.5)}
    ridge_model=spend_models["ridge"]
    # Lasso exact-zero coefficients after standardisation.
    lasso_coefs=dict(zip(SPEND_FEATURES,spend_models["lasso"].named_steps["model"].coef_))
    zero_features=[f for f,v in lasso_coefs.items() if v==0]

    # Coefficient stability: five 80% subsamples of the spend-model training rows.
    coef_values={"linear":[],"ridge_alpha_30":[]}
    for seed in range(5):
        sample=X_train.sample(frac=.8,random_state=seed)
        ys=y_train.loc[sample.index]
        lm=Pipeline([("scale",StandardScaler()),("model",LinearRegression())]).fit(sample,ys)
        rm=Pipeline([("scale",StandardScaler()),("model",Ridge(alpha=30))]).fit(sample,ys)
        idx=SPEND_FEATURES.index("hours_watched_per_week")
        coef_values["linear"].append(float(lm.named_steps["model"].coef_[idx]))
        coef_values["ridge_alpha_30"].append(float(rm.named_steps["model"].coef_[idx]))
    stability={k:{"coefficients":v,"population_std_ddof0":float(np.std(v,ddof=0))} for k,v in coef_values.items()}

    # Predict next-cycle recharge for every Part B test subscriber, including churners.
    scores=pd.read_csv(OUT/"test_scores.csv")
    test_rows=s.set_index("subscriber_id").loc[scores.subscriber_id,SPEND_FEATURES]
    predicted_spend=ridge_model.predict(test_rows)
    risk=scores.copy(); risk["predicted_next_recharge_inr"]=predicted_spend
    risk["revenue_at_risk_inr"]=risk.churn_probability*risk.predicted_next_recharge_inr
    risk=risk.merge(s[["subscriber_id","segment_name"]],on="subscriber_id",how="left")
    risk=risk[["subscriber_id","churn_probability","predicted_next_recharge_inr","revenue_at_risk_inr","segment_name","churned"]]
    risk=risk.sort_values("revenue_at_risk_inr",ascending=False).head(15)
    risk.to_csv(OUT/"retention_priority.csv",index=False)

    metrics={"k_scores":ks,"chosen_k":chosen_k,"segment_profile":profile.to_dict(orient="records"),
      "new_subscriber_assignment":{"nearest_segment_id":nearest,"segment_name":names[nearest],"distances":distances.tolist(),"predict":predicted},
      "correlation_matrix":corr.to_dict(),"correlation_pairs_abs_gt_0_9":pairs,
      "spend_split":{"train_rows":len(X_train),"test_rows":len(X_test)},"spend_model":spend_metrics,"lasso_coefficients":{k:float(v) for k,v in lasso_coefs.items()},"lasso_zero_features":zero_features,
      "coefficient_stability":stability,"priority":{"count":len(risk),"sum_revenue_at_risk_inr":float(risk.revenue_at_risk_inr.sum()),"actual_churners":int(risk.churned.sum())}}
    (OUT/"part_c_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": run()
