"""Retention brief generation for the top-priority subscribers."""
from pathlib import Path
import json, sqlite3
import pandas as pd, numpy as np, joblib
from pydantic import BaseModel
from typing import Literal
from llm_client import call_llm
from part_d import build_collection,retrieve
from features import build_features,NUMERIC,CATEGORICAL

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"

class RetentionBrief(BaseModel):
    """Schema for an auditable retention recommendation."""
    subscriber_id:str
    risk_band:Literal["High","Medium","Low"]
    recommended_action:str
    policy_source:str

def build_brief(subscriber_id, collection, priority):
    """Build, validate and return a retention brief for one subscriber."""
    db=sqlite3.connect(ROOT/"data/tataplay.db")
    s=pd.read_sql_query("SELECT * FROM subscribers WHERE subscriber_id=?",db,params=(subscriber_id,))
    r=pd.read_sql_query("SELECT * FROM recharges WHERE subscriber_id=?",db,params=(subscriber_id,))
    db.close()
    features=build_features(s,r).replace({None:np.nan})
    model=joblib.load(OUT/"churn_model.joblib")
    scored=float(model.predict_proba(features[NUMERIC+CATEGORICAL])[:,1][0])
    saved=float(priority.loc[priority.subscriber_id==subscriber_id,"churn_probability"].iloc[0])
    assert round(scored,4)==round(saved,4)
    band="High" if saved>=.4 else ("Medium" if saved>=.2 else "Low")
    seg=pd.read_csv(OUT/"segments.csv").set_index("subscriber_id").loc[subscriber_id,"segment_name"]
    tenure=int(s.tenure_months.iloc[0]); failed=int(features.n_failed_6m.iloc[0])
    hits=retrieve(collection,f"retention offer for a {band} risk subscriber with {tenure} months tenure and {failed} failed recharges")
    ctx="\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user=(f"FACTS:\nsubscriber_id: {subscriber_id}\nrisk_band: {band}\ntenure_months: {tenure}\n"
          f"failed_recharges_6m: {failed}\nsegment: {seg}\n\nPOLICY CONTEXT:\n{ctx}")
    system=("RETENTION BRIEF Reply with one JSON object only. Use only the policy context. "
            "Follow the retention offer eligibility rules.")
    reply=call_llm([{"role":"system","content":system},{"role":"user","content":user}])
    brief=RetentionBrief.model_validate_json(reply)
    assert brief.subscriber_id==subscriber_id and brief.risk_band==band
    assert brief.policy_source in {p.name for p in (ROOT/"data/policies").glob("*.txt")}
    return {"subscriber_id":subscriber_id,"churn_probability":saved,"risk_band":band,
            "tenure_months":tenure,"failed_recharges_6m":failed,"segment":seg,
            "brief":brief.model_dump()}

def run():
    """Generate briefs for every row in the retention priority file."""
    priority=pd.read_csv(OUT/"retention_priority.csv")
    _,col=build_collection()
    results=[build_brief(sid,col,priority) for sid in priority.subscriber_id]
    with (OUT/"retention_briefs.jsonl").open("w") as f:
        for row in results: f.write(json.dumps(row)+"\n")
    print("BRIEFS",len(results),pd.Series([r["risk_band"] for r in results]).value_counts().to_dict())

if __name__=="__main__": run()
