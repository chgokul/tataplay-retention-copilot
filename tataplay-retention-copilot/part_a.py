"""Part A: audit, clean, persist and execute SQL."""
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW, CLEAN = ROOT/"data/raw", ROOT/"data/clean"
CLEAN.mkdir(parents=True, exist_ok=True)

def run():
    """Run Part A and save clean CSVs plus SQLite database."""
    subs = pd.read_csv(RAW/"subscribers.csv")
    rech = pd.read_csv(RAW/"recharges.csv")
    plans = pd.read_csv(RAW/"plans.csv")
    print("RAW SUBSCRIBERS", subs.shape, subs.dtypes.to_dict())
    print("SUBSCRIBER MISSING", pd.DataFrame({"count":subs.isna().sum(),"pct":subs.isna().mean()*100}))
    print("RAW RECHARGES", rech.shape, rech.dtypes.to_dict())
    print("RECHARGE MISSING", pd.DataFrame({"count":rech.isna().sum(),"pct":rech.isna().mean()*100}))
    print("AUDIT", {"subscriber_duplicates":int(subs.duplicated().sum()),
          "subscriber_ids":int(subs.subscriber_id.nunique()),
          "recharge_duplicates":int(rech.duplicated().sum()),
          "recharge_ids":int(rech.recharge_id.nunique())})
    subs = subs.drop_duplicates().copy()
    city_map = {"bangalore":"Bengaluru","bombay":"Mumbai","madras":"Chennai",
                "calcutta":"Kolkata","new delhi":"Delhi","hyd":"Hyderabad"}
    subs["city"] = subs.city.astype(str).str.strip().str.lower().map(
        lambda x: city_map.get(x, x.title()))
    subs["city"] = subs.city.replace({"Bengaluru":"Bengaluru"})
    subs["auto_renew"] = subs.auto_renew.astype(str).str.strip().str.lower().map(
        {"yes":"Yes","y":"Yes","no":"No","n":"No"})
    subs["age"] = pd.to_numeric(subs.age, errors="coerce")
    subs.loc[~subs.age.between(10,100), "age"] = pd.NA
    rech = rech.drop_duplicates().copy()
    rech = rech[rech.subscriber_id.isin(subs.subscriber_id)].copy()
    rech["status"] = rech.status.astype(str).str.strip().str.lower()
    rech["amount_inr"] = pd.to_numeric(rech.amount_inr, errors="coerce")
    rebuild = rech.amount_inr.isna()
    monthly = subs.set_index("subscriber_id").monthly_recharge_inr
    rech.loc[rebuild, "amount_inr"] = rech.loc[rebuild, "subscriber_id"].map(monthly)
    rech["recharge_date"] = pd.to_datetime(rech.recharge_date).dt.strftime("%Y-%m-%d")
    for name, frame in [("subscribers",subs),("recharges",rech),("plans",plans)]:
        frame.to_csv(CLEAN/f"{name}.csv", index=False)
    db = ROOT/"data/tataplay.db"
    db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db) as con:
        subs.to_sql("subscribers", con, if_exists="replace", index=False)
        rech.to_sql("recharges", con, if_exists="replace", index=False)
        plans.to_sql("plans", con, if_exists="replace", index=False)
    print("CLEAN", {"subscribers":len(subs),"recharges":len(rech),
          "rebuilt_amounts":int(rebuild.sum()),"churners":int(subs.churned.sum())})
    sql = (ROOT/"sql/queries.sql").read_text()
    with sqlite3.connect(db) as con:
        for i, statement in enumerate([x.strip() for x in sql.split(";") if x.strip()],1):
            print(f"SQL Q{i}\n", pd.read_sql_query(statement, con).to_string(index=False))

if __name__ == "__main__":
    run()
