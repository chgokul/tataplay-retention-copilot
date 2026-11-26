"""Part B: leakage-safe churn modelling, threshold selection and scoring."""
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from features import build_features, NUMERIC, CATEGORICAL

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)

def cost_at_threshold(y, prob, threshold):
    """Calculate the business cost of missed churners and unnecessary offers."""
    pred=(prob>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return 1200*fn+300*fp, (tn,fp,fn,tp)

def run():
    """Train, tune, select by validation cost, then evaluate the test set once."""
    s=pd.read_csv(ROOT/"data/clean/subscribers.csv")
    r=pd.read_csv(ROOT/"data/clean/recharges.csv")
    df=build_features(s,r)
    X=df[NUMERIC+CATEGORICAL].copy()
    y=df.churned.astype(int)
    X_trainval,X_test,y_trainval,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    X_train,X_val,y_train,y_val=train_test_split(X_trainval,y_trainval,test_size=.25,stratify=y_trainval,random_state=42)
    pre=ColumnTransformer([
      ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),NUMERIC),
      ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),CATEGORICAL)])
    models={
      "logistic":(Pipeline([("pre",pre),("model",LogisticRegression(max_iter=1000))]),
        {"model__C":[.01,.1,1,10,100]}),
      "random_forest":(Pipeline([("pre",pre),("model",RandomForestClassifier(random_state=42))]),
        {"model__n_estimators":[50,100,150],"model__max_depth":[3,4,5,6,8],"model__max_features":[3,4,5]})
    }
    best={}
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for name,(pipe,grid) in models.items():
        gs=GridSearchCV(pipe,grid,scoring="f1",cv=cv,n_jobs=-1,return_train_score=True)
        gs.fit(X_train,y_train)
        best[name]=gs.best_estimator_
        print(name,"best",gs.best_params_,"cv_f1",gs.best_score_,
              "std",gs.cv_results_["std_test_score"][gs.best_index_])
    selected=None; chosen_threshold=None; chosen_cost=float("inf")
    for name,model in best.items():
        pv=model.predict_proba(X_val)[:,1]
        costs=[(cost_at_threshold(y_val,pv,t)[0],t) for t in np.arange(.10,.901,.05)]
        c,t=min(costs)
        print(name,"validation_cost",c,"threshold",round(float(t),2))
        if c<chosen_cost: selected,chosen_threshold,chosen_cost=model,float(t),c
    # Test is used only after model and threshold have been selected.
    p=selected.predict_proba(X_test)[:,1]
    cost,counts=cost_at_threshold(y_test,p,chosen_threshold)
    pred=(p>=chosen_threshold).astype(int)
    print("TEST",{"threshold":chosen_threshold,"counts":counts,"cost":cost,
      "precision":precision_score(y_test,pred,zero_division=0),
      "recall":recall_score(y_test,pred,zero_division=0),
      "f1":f1_score(y_test,pred,zero_division=0)})
    joblib.dump(selected,OUT/"churn_model.joblib")
    test_ids=df.loc[X_test.index,"subscriber_id"]
    pd.DataFrame({"subscriber_id":test_ids,"churn_probability":p,"churned":y_test.values}).to_csv(OUT/"test_scores.csv",index=False)
    # Validation-selected threshold is persisted for inference.
    (OUT/"threshold.txt").write_text(str(chosen_threshold))
    print("SPLITS",len(X_train),len(X_val),len(X_test),"CHURN",y_train.sum(),y_val.sum(),y_test.sum())

if __name__=="__main__": run()
