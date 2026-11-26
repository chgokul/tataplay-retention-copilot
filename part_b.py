"""Part B: complete leakage-safe churn experiments and cost-based model selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)
from features import build_features, NUMERIC, CATEGORICAL

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)

def cost_at_threshold(y, prob, threshold):
    """Return INR cost and (TN, FP, FN, TP) at a probability threshold."""
    pred=(np.asarray(prob)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return int(1200*fn+300*fp),(int(tn),int(fp),int(fn),int(tp))

def make_preprocessor():
    """Create the numeric and categorical preprocessing transformer."""
    return ColumnTransformer([
      ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),NUMERIC),
      ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),CATEGORICAL)])

def evaluate_model(model, X, y, threshold):
    """Calculate confusion counts, classification metrics and business cost."""
    p=model.predict_proba(X)[:,1]
    cost,counts=cost_at_threshold(y,p,threshold)
    pred=(p>=threshold).astype(int)
    return {"counts_tn_fp_fn_tp":counts,"cost_inr":cost,
      "precision":precision_score(y,pred,zero_division=0),
      "recall":recall_score(y,pred,zero_division=0),
      "f1":f1_score(y,pred,zero_division=0),"probabilities":p}

def run():
    """Run all Part B experiments and persist the selected fitted pipeline."""
    s=pd.read_csv(ROOT/"data/clean/subscribers.csv")
    r=pd.read_csv(ROOT/"data/clean/recharges.csv")
    df=build_features(s,r)
    X=df[NUMERIC+CATEGORICAL].copy(); y=df.churned.astype(int)
    X_trainval,X_test,y_trainval,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    X_train,X_val,y_train,y_val=train_test_split(X_trainval,y_trainval,test_size=.25,stratify=y_trainval,random_state=42)
    metrics={"class_balance":{"churners":int(y.sum()),"total":len(y),"rate":float(y.mean())},
      "split_sizes":{"train":len(X_train),"validation":len(X_val),"test":len(X_test)},
      "split_churners":{"train":int(y_train.sum()),"validation":int(y_val.sum()),"test":int(y_test.sum())},
      "feature_columns":{"numeric":NUMERIC,"categorical":CATEGORICAL},
      "recency_missing":int(df.recency_days.isna().sum())}

    # Leakage experiment: use the same pipeline shape, with and without post-outcome columns.
    leak_num=NUMERIC+["next_cycle_recharge_inr"]
    leak_cat=CATEGORICAL+["closure_request_logged"]
    def leak_pipe(nums,cats):
        pre=ColumnTransformer([
          ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
          ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats)])
        return Pipeline([("pre",pre),("model",RandomForestClassifier(n_estimators=100,random_state=42))])
    leak_results={}
    for name,nums,cats in [("with_leakage",leak_num,leak_cat),("without_leakage",NUMERIC,CATEGORICAL)]:
        cols=nums+cats
        xx=df[cols].copy()
        a,b=xx.loc[X_train.index],xx.loc[X_val.index]
        model=leak_pipe(nums,cats).fit(a,y_train)
        pp=model.predict(b); leak_results[name]={"accuracy":accuracy_score(y_val,pp),"churn_f1":f1_score(y_val,pp,zero_division=0)}
    metrics["leakage_experiment"]=leak_results

    # Fit the actual preprocessing only on training rows and record learned statistics.
    pre=make_preprocessor(); transformed=pre.fit_transform(X_train)
    metrics["preprocessing"]={"output_columns":int(transformed.shape[1]),
      "median_age":float(pre.named_transformers_["num"].named_steps["imputer"].statistics_[NUMERIC.index("age")]),
      "scaled_tenure_mean":float(pre.named_transformers_["num"].named_steps["scale"].mean_[NUMERIC.index("tenure_months")]),
      "full_data_tenure_mean":float(X.tenure_months.mean()),
      "no_missing_after_transform":bool(np.isfinite(transformed.toarray() if hasattr(transformed,"toarray") else transformed).all())}

    # Baselines and simple models.
    majority=np.zeros(len(y_val),dtype=int)
    metrics["majority_baseline"]={"accuracy":accuracy_score(y_val,majority),"churn_f1":f1_score(y_val,majority,zero_division=0)}
    logit=Pipeline([("pre",make_preprocessor()),("model",LogisticRegression(max_iter=1000))]).fit(X_train,y_train)
    lp=logit.predict(X_val)
    metrics["logistic_baseline"]={"accuracy":accuracy_score(y_val,lp),"precision":precision_score(y_val,lp,zero_division=0),
       "recall":recall_score(y_val,lp,zero_division=0),"f1":f1_score(y_val,lp,zero_division=0)}
    tree=Pipeline([("pre",make_preprocessor()),("model",DecisionTreeClassifier(random_state=42))]).fit(X_train,y_train)
    rf_default=Pipeline([("pre",make_preprocessor()),("model",RandomForestClassifier(n_estimators=100,random_state=42))]).fit(X_train,y_train)
    metrics["overfit_models"]={"decision_tree":{"train_accuracy":accuracy_score(y_train,tree.predict(X_train)),"validation_f1":f1_score(y_val,tree.predict(X_val))},
      "random_forest":{"train_accuracy":accuracy_score(y_train,rf_default.predict(X_train)),"validation_f1":f1_score(y_val,rf_default.predict(X_val))}}
    depth=[]
    for d in range(1,11):
        m=Pipeline([("pre",make_preprocessor()),("model",DecisionTreeClassifier(max_depth=d,random_state=42))]).fit(X_train,y_train)
        depth.append({"max_depth":d,"train_accuracy":accuracy_score(y_train,m.predict(X_train)),"validation_f1":f1_score(y_val,m.predict(X_val))})
    metrics["tree_depth_sweep"]=depth

    # Five-fold CV grid searches on training rows only.
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    grids={
      "logistic":(Pipeline([("pre",make_preprocessor()),("model",LogisticRegression(max_iter=1000))]),
                  {"model__C":[.01,.1,1,10,100]}),
      "random_forest":(Pipeline([("pre",make_preprocessor()),("model",RandomForestClassifier(random_state=42))]),
                  {"model__n_estimators":[50,100,150],"model__max_depth":[3,4,5,6,8],"model__max_features":[3,4,5]})}
    tuned={}; cv_results={}
    for name,(pipe,grid) in grids.items():
        gs=GridSearchCV(pipe,grid,scoring="f1",cv=cv,n_jobs=-1,return_train_score=True)
        gs.fit(X_train,y_train); tuned[name]=gs.best_estimator_
        cv_results[name]={"best_params":gs.best_params_,"best_mean_f1":float(gs.best_score_),
          "std_f1":float(gs.cv_results_["std_test_score"][gs.best_index_]),"parameter_combinations":len(gs.cv_results_["params"]),
          "model_fits":len(gs.cv_results_["params"])*5}
    metrics["cross_validation"]=cv_results

    # Thresholds are selected exclusively on validation; test is touched afterwards.
    thresholds={}; validation_costs={}
    for name,m in tuned.items():
        pv=m.predict_proba(X_val)[:,1]
        choices=[(cost_at_threshold(y_val,pv,float(t))[0],float(t)) for t in np.round(np.arange(.10,.901,.05),2)]
        best_cost,best_t=min(choices,key=lambda x:(x[0],x[1]))
        thresholds[name]=best_t; validation_costs[name]={"threshold":best_t,"cost_inr":best_cost}
    selected_name=min(validation_costs,key=lambda n:validation_costs[n]["cost_inr"])
    selected=tuned[selected_name]; threshold=thresholds[selected_name]
    metrics["validation_thresholds"]=validation_costs
    metrics["selected_model"]=selected_name
    test_results={}
    for name,m in tuned.items():
        ev=evaluate_model(m,X_test,y_test,thresholds[name])
        test_results[name]={k:v for k,v in ev.items() if k!="probabilities"}
    metrics["test_results"]=test_results
    metrics["reference_policies"]={"no_action_cost_inr":1200*int(y_test.sum()),"offer_everyone_cost_inr":300*int((y_test==0).sum())+0*int(y_test.sum())+1200*0}
    # The specified offer-everyone policy incurs 300 for each true non-churner.
    metrics["reference_policies"]["offer_everyone_cost_inr"]=300*int((y_test==0).sum())

    # Feature importance and logistic coefficient signs/names.
    rf=tuned["random_forest"]
    names=rf.named_steps["pre"].get_feature_names_out()
    importances=rf.named_steps["model"].feature_importances_
    order=np.argsort(importances)[::-1][:5]
    metrics["random_forest_top_features"]=[{"feature":str(names[i]),"importance":float(importances[i])} for i in order]
    lm=tuned["logistic"].named_steps["model"]
    lnames=tuned["logistic"].named_steps["pre"].get_feature_names_out()
    coef=dict(zip(lnames,lm.coef_[0]))
    metrics["logistic_coefficients"]={k:float(coef[k]) for k in ["num__support_calls_90d","num__tenure_months","num__n_failed_6m"]}

    # Persist selected model, prove reload parity and write the 480-row test score file.
    joblib.dump(selected,OUT/"churn_model.joblib")
    loaded=joblib.load(OUT/"churn_model.joblib")
    original=selected.predict_proba(X_test)[:,1]; reloaded=loaded.predict_proba(X_test)[:,1]
    metrics["reload_predictions_identical"]=bool(np.array_equal(original,reloaded))
    scores=pd.DataFrame({"subscriber_id":df.loc[X_test.index,"subscriber_id"].values,
                         "churn_probability":original,"churned":y_test.values})
    scores.to_csv(OUT/"test_scores.csv",index=False)
    (OUT/"threshold.txt").write_text(str(threshold))
    metrics["selected_threshold"]=threshold
    (OUT/"part_b_metrics.json").write_text(json.dumps(metrics,indent=2,default=lambda x:float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else x))
    print(json.dumps(metrics,indent=2,default=str))

if __name__=="__main__": run()
