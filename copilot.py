"""Generate policy-grounded retention briefs for the 15 highest-risk subscribers."""
from pathlib import Path
import json, sqlite3
import pandas as pd, numpy as np, joblib
from pydantic import BaseModel
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from llm_client import call_llm
from part_d import build_collection,retrieve
from features import build_features,NUMERIC,CATEGORICAL

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"; POLICY_DIR=ROOT/"data/policies"

class RetentionBrief(BaseModel):
    """Schema for the final validated retention recommendation."""
    subscriber_id:str
    risk_band:Literal["High","Medium","Low"]
    recommended_action:str
    policy_source:str

def build_brief(subscriber_id, collection=None, priority=None):
    """Build, re-score, retrieve, validate and return one subscriber's retention brief."""
    if priority is None: priority=pd.read_csv(OUT/"retention_priority.csv")
    if collection is None: _,collection,_=build_collection()
    db=sqlite3.connect(ROOT/"data/tataplay.db")
    try:
        sub=pd.read_sql_query("SELECT * FROM subscribers WHERE subscriber_id=?",db,params=(subscriber_id,))
        recharges=pd.read_sql_query("SELECT * FROM recharges WHERE subscriber_id=?",db,params=(subscriber_id,))
        # Required SQL-derived six-month failed/success counts; feature builder independently derives the full feature row.
        counts=pd.read_sql_query("SELECT SUM(CASE WHEN LOWER(status)='failed' THEN 1 ELSE 0 END) AS failed_count, SUM(CASE WHEN LOWER(status)='success' THEN 1 ELSE 0 END) AS success_count FROM recharges WHERE subscriber_id=? AND recharge_date <= '2026-03-31'",db,params=(subscriber_id,)).iloc[0]
    finally: db.close()
    if sub.empty: raise ValueError(f"Unknown subscriber_id: {subscriber_id}")
    features=build_features(sub,recharges).replace({None:np.nan})
    model=joblib.load(OUT/"churn_model.joblib")
    rescored=float(model.predict_proba(features[NUMERIC+CATEGORICAL])[:,1][0])
    row=priority.loc[priority.subscriber_id==subscriber_id]
    if row.empty: raise ValueError(f"Subscriber {subscriber_id} is not in retention_priority.csv")
    saved=float(row.churn_probability.iloc[0])
    if round(rescored,4)!=round(saved,4): raise AssertionError(f"Probability mismatch for {subscriber_id}: {rescored} vs {saved}")
    band="High" if saved>=.40 else ("Medium" if saved>=.20 else "Low")
    segments=pd.read_csv(OUT/"segments.csv").set_index("subscriber_id")
    segment=str(segments.loc[subscriber_id,"segment_name"])
    tenure=int(sub.tenure_months.iloc[0]); failed=int(counts.failed_count or 0)
    hits=retrieve(collection,f"retention offer for a {band} risk subscriber with {tenure} months tenure and {failed} failed recharges",k=3)
    context="\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user=(f"FACTS:\nsubscriber_id: {subscriber_id}\nrisk_band: {band}\ntenure_months: {tenure}\n"
          f"failed_recharges_6m: {failed}\nsegment: {segment}\n\nPOLICY CONTEXT:\n{context}")
    system="RETENTION BRIEF Reply with one JSON object only. Use only the policy context. Follow the retention offer eligibility rules."
    parser=PydanticOutputParser(pydantic_object=RetentionBrief)
    messages=[{"role":"system","content":system+"\n"+parser.get_format_instructions()},
              {"role":"user","content":user}]
    record=None; attempts=0; errors=[]
    for attempt in range(3):
        attempts+=1; reply=call_llm(messages)
        try:
            record=parser.parse(reply); break
        except Exception as exc:
            errors.append(str(exc))
            if attempt==2: raise
            messages.extend([{"role":"assistant","content":reply},
                             {"role":"user","content":"VALIDATION ERROR: "+str(exc)}])
    if record.subscriber_id!=subscriber_id or record.risk_band!=band:
        raise AssertionError("Retention brief subscriber_id or risk_band did not match input")
    if record.policy_source not in {p.name for p in POLICY_DIR.glob("*.txt")}:
        raise AssertionError("Retention brief references a non-existent policy file")
    if record.policy_source!="retention_offers_policy.txt":
        raise AssertionError("Retention brief must cite retention_offers_policy.txt")
    return {"subscriber_id":subscriber_id,"churn_probability":saved,"risk_band":band,
      "tenure_months":tenure,"failed_recharges_6m":failed,"segment":segment,
      "attempts":attempts,"brief":record.model_dump()}

def run():
    """Generate and persist one validated JSONL brief for each priority subscriber."""
    priority=pd.read_csv(OUT/"retention_priority.csv")
    _,collection,_=build_collection()
    results=[build_brief(str(sid),collection,priority) for sid in priority.subscriber_id]
    with (OUT/"retention_briefs.jsonl").open("w",encoding="utf-8") as f:
        for row in results: f.write(json.dumps(row)+"\n")
    action_counts=pd.Series([r["brief"]["recommended_action"] for r in results]).value_counts().to_dict()
    risk_counts=pd.Series([r["risk_band"] for r in results]).value_counts().to_dict()
    metrics={"valid_briefs":len(results),"risk_band_counts":risk_counts,"recommended_action_counts":action_counts}
    (OUT/"copilot_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": run()
