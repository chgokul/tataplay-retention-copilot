"""Part A: audit, clean, persist to SQLite and execute the four SQL analyses."""
from pathlib import Path
import json, sqlite3
import pandas as pd

ROOT=Path(__file__).resolve().parent; RAW=ROOT/"data/raw"; CLEAN=ROOT/"data/clean"; OUT=ROOT/"outputs"
CLEAN.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True)

def audit_table(frame):
    """Return shape, dtypes and per-column missing counts and percentages."""
    missing=pd.DataFrame({"missing_count":frame.isna().sum(),"missing_pct":frame.isna().mean()*100})
    return {"shape":list(frame.shape),"dtypes":{k:str(v) for k,v in frame.dtypes.items()},
            "missing":missing.round(4).to_dict(orient="index")}

def run():
    """Run the complete Part A audit and produce clean CSV and SQLite tables."""
    subs_raw=pd.read_csv(RAW/"subscribers.csv"); rech_raw=pd.read_csv(RAW/"recharges.csv"); plans=pd.read_csv(RAW/"plans.csv")
    subs_audit=audit_table(subs_raw); rech_audit=audit_table(rech_raw)
    subscriber_audit={"rows":len(subs_raw),"exact_duplicates":int(subs_raw.duplicated().sum()),
      "distinct_subscriber_id":int(subs_raw.subscriber_id.nunique()),"blank_age":int(subs_raw.age.isna().sum()),
      "age_outside_10_100":int((~pd.to_numeric(subs_raw.age,errors="coerce").between(10,100)).sum()-subs_raw.age.isna().sum()),
      "blank_payment_mode":int(subs_raw.payment_mode.isna().sum()),"distinct_raw_city_spellings":int(subs_raw.city.nunique()),
      "distinct_raw_auto_renew_spellings":int(subs_raw.auto_renew.nunique()),
      "blank_next_cycle_recharge":int(subs_raw.next_cycle_recharge_inr.isna().sum())}
    numeric_amount=pd.to_numeric(rech_raw.amount_inr,errors="coerce")
    recharge_audit={"rows":len(rech_raw),"exact_duplicates":int(rech_raw.duplicated().sum()),
      "distinct_recharge_id":int(rech_raw.recharge_id.nunique()),"non_numeric_amounts":int(numeric_amount.isna().sum()),
      "distinct_raw_status_spellings":int(rech_raw.status.nunique()),
      "orphan_subscriber_rows":int((~rech_raw.subscriber_id.isin(subs_raw.subscriber_id)).sum())}
    print("RAW SUBSCRIBER AUDIT",json.dumps(subscriber_audit))
    print("RAW RECHARGE AUDIT",json.dumps(recharge_audit))
    print("SUBSCRIBER COLUMN AUDIT",json.dumps(subs_audit,indent=2))
    print("RECHARGE COLUMN AUDIT",json.dumps(rech_audit,indent=2))

    # Remove exact duplicates; normalise only the fields required by the brief.
    subs=subs_raw.drop_duplicates().copy()
    city_map={"bangalore":"Bengaluru","bombay":"Mumbai","madras":"Chennai","calcutta":"Kolkata","new delhi":"Delhi","hyd":"Hyderabad"}
    subs["city"]=subs.city.astype(str).str.strip().str.lower().map(lambda x:city_map.get(x,x.title()))
    subs["auto_renew"]=subs.auto_renew.astype(str).str.strip().str.lower().map({"yes":"Yes","y":"Yes","no":"No","n":"No"})
    subs["age"]=pd.to_numeric(subs.age,errors="coerce")
    invalid_age=~subs.age.between(10,100)
    invalid_age_count=int((subs.age.notna() & invalid_age).sum())
    subs.loc[invalid_age,"age"]=pd.NA

    rech=rech_raw.drop_duplicates().copy()
    before_orphan=len(rech); rech=rech[rech.subscriber_id.isin(subs.subscriber_id)].copy(); orphan_removed=before_orphan-len(rech)
    rech["status"]=rech.status.astype(str).str.strip().str.lower()
    rech["amount_inr"]=pd.to_numeric(rech.amount_inr,errors="coerce")
    rebuild=rech.amount_inr.isna(); rebuilt_count=int(rebuild.sum())
    monthly=subs.set_index("subscriber_id").monthly_recharge_inr
    rech.loc[rebuild,"amount_inr"]=rech.loc[rebuild,"subscriber_id"].map(monthly)
    rech["recharge_date"]=pd.to_datetime(rech.recharge_date).dt.strftime("%Y-%m-%d")
    for name,frame in [("subscribers",subs),("recharges",rech),("plans",plans)]:
        frame.to_csv(CLEAN/f"{name}.csv",index=False)
    db=ROOT/"data/tataplay.db"
    with sqlite3.connect(db) as con:
        subs.to_sql("subscribers",con,if_exists="replace",index=False)
        rech.to_sql("recharges",con,if_exists="replace",index=False)
        plans.to_sql("plans",con,if_exists="replace",index=False)

    metrics={"raw_subscribers":{"column_audit":subs_audit,"summary":subscriber_audit},
      "raw_recharges":{"column_audit":rech_audit,"summary":recharge_audit},
      "cleaning":{"subscribers_remaining":len(subs),"recharges_remaining":len(rech),"orphan_recharge_rows_removed":orphan_removed,
        "invalid_ages_set_missing":invalid_age_count,"recharge_amounts_rebuilt":rebuilt_count,
        "city_counts":subs.city.value_counts().to_dict(),"auto_renew_counts":subs.auto_renew.value_counts().to_dict(),
        "missing_age_after_cleaning":int(subs.age.isna().sum()),"missing_payment_mode_after_cleaning":int(subs.payment_mode.isna().sum()),
        "churned":int(subs.churned.sum()),"churn_rate":float(subs.churned.mean()),
        "recharge_status_counts":rech.status.value_counts().to_dict(),
        "successful_recharge_amount_inr":float(rech.loc[rech.status=="success","amount_inr"].sum())}}
    print("CLEANING SUMMARY",json.dumps(metrics["cleaning"],indent=2))

    sql=(ROOT/"sql/queries.sql").read_text(encoding="utf-8")
    statements=[x.strip() for x in sql.split(";") if x.strip()]
    with sqlite3.connect(db) as con:
        q1=pd.read_sql_query(statements[0],con)
        q2=pd.read_sql_query(statements[1],con)
        q3=pd.read_sql_query(statements[2],con)
        q4=pd.read_sql_query(statements[3],con)
        # Independent pandas groupby check for the Premium row in Q1.
        check=(rech[rech.status=="success"].merge(subs[["subscriber_id","plan_id"]],on="subscriber_id")
          .merge(plans[["plan_id","plan_name"]],on="plan_id").groupby("plan_name").amount_inr.agg(["sum","mean"]))
        print("SQL Q1\n",q1.to_string(index=False))
        print("PANDAS GROUPBY CHECK (Premium)\n",check.loc["Premium"].to_dict())
        print("SQL Q2 count",len(q2),"zero-tenure",int((q2.tenure_months==0).sum()))
        rank_counts=q3[q3.rnk==1].groupby("city").size().to_dict()
        kolkata=q3[q3.city=="Kolkata"].head(4)
        print("SQL Q3 RANK=1 COUNTS",rank_counts)
        print("SQL Q3 KOLKATA TOP FOUR\n",kolkata.to_string(index=False))
        print("SQL Q4\n",q4.to_string(index=False))
        metrics["sql"]={"q1":q1.to_dict(orient="records"),"pandas_premium_check":{k:float(v) for k,v in check.loc["Premium"].to_dict().items()},
          "q2_count":len(q2),"q2_zero_tenure":int((q2.tenure_months==0).sum()),"q3_rank1_counts":rank_counts,
          "q3_kolkata_top_four":kolkata.to_dict(orient="records"),"q4":q4.to_dict(orient="records")}
    (OUT/"part_a_metrics.json").write_text(json.dumps(metrics,indent=2,default=str))

if __name__=="__main__": run()
