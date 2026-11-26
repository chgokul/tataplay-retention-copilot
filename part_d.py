"""Part D: offline ticket workflows, validated extraction and policy RAG."""
from pathlib import Path
import json, re, csv
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from llm_client import call_llm
import llm_client
import mock_llm

ROOT=Path(__file__).resolve().parent; RAW=ROOT/"data/raw"; POL=ROOT/"data/policies"; OUT=ROOT/"outputs"
LABELS=("billing","signal","plan_change","cancellation")
FALLBACK="I don't have enough information to answer that."

class TicketRecord(BaseModel):
    """Validated fields extracted from one customer support ticket."""
    subscriber_id: Optional[str]=None
    category: Literal["billing","signal","plan_change","cancellation","other"]
    urgency: int=Field(ge=1,le=5)
    wants_refund: bool

class RetentionBrief(BaseModel):
    """Validated retention action with its policy source."""
    subscriber_id: str
    risk_band: Literal["High","Medium","Low"]
    recommended_action: str
    policy_source: str

def invoke_prompt_value(prompt_value):
    """Convert LangChain messages to chat dictionaries and call the offline client."""
    messages=[]
    for message in prompt_value.to_messages():
        role={"system":"system","human":"user","ai":"assistant"}.get(message.type,"user")
        messages.append({"role":role,"content":message.content})
    return call_llm(messages)

def pydantic_smoke_tests():
    """Demonstrate Pydantic coercion and field validation errors required by the brief."""
    results={}
    converted=TicketRecord(category="billing",urgency="3",wants_refund=False)
    results["string_3_converted_to"]={"value":converted.urgency,"type":type(converted.urgency).__name__}
    for name,payload in [("urgency_9",{"category":"billing","urgency":9,"wants_refund":False}),
                         ("category_refund",{"category":"refund","urgency":2,"wants_refund":True})]:
        try: TicketRecord(**payload)
        except ValidationError as exc:
            results[name]=[{"field":".".join(str(x) for x in e["loc"]),"error_type":e["type"]} for e in exc.errors()]
    return results

def run_classifier():
    """Run weak, zero-shot and few-shot classification on the 146 readable tickets."""
    tickets=pd.read_csv(RAW/"tickets.csv"); tickets=tickets[tickets.text!="..."].reset_index(drop=True)
    examples=pd.read_csv(RAW/"few_shot_examples.csv")
    strong=("TICKET CLASSIFIER You are a support ticket classifier. Classify the customer's primary intent into exactly one of these labels: "
            "billing, signal, plan_change, cancellation. Respond with only the label. Do not create new labels.")
    few=strong+"\n"+"\n".join(f"Ticket: {r.text}\nLabel: {r.label}" for r in examples.itertuples())
    outputs={}; stats={}
    for name,system in [("weak","TICKET CLASSIFIER. Classify the ticket."),("zero_shot",strong),("few_shot",few)]:
        mock_llm.reset(); llm_client.RATE_LIMIT_RETRIES=0
        prompt=ChatPromptTemplate.from_messages([("system",system),("human","{ticket}")])
        chain=prompt|invoke_prompt_value|StrOutputParser()
        predictions=[chain.invoke({"ticket":t.text}) for t in tickets.itertuples()]
        valid=sum(x in LABELS for x in predictions)
        correct=sum(a==b for a,b in zip(predictions,tickets.true_category))
        stats[name]={"valid_labels":valid,"correct":correct,"accuracy":correct/len(tickets),
                     "recovered_rate_limits":llm_client.RATE_LIMIT_RETRIES,"mock_post_calls":mock_llm._state["calls"]}
        outputs[name]=predictions
    result=pd.DataFrame({"ticket_id":tickets.ticket_id,"true_category":tickets.true_category,
                         "zero_shot":outputs["zero_shot"],"few_shot":outputs["few_shot"]})
    result.to_csv(OUT/"ticket_classification.csv",index=False)
    wrong=result[result.true_category!=result.few_shot]
    return stats,wrong.to_dict(orient="records")

def run_extraction():
    """Extract all 150 tickets with a LangChain parser chain, two retries and manual review."""
    parser=PydanticOutputParser(pydantic_object=TicketRecord)
    prompt=ChatPromptTemplate.from_messages([
      ("system","TICKET EXTRACTOR Extract the ticket fields.\n{format_instructions}"),
      ("human","{ticket}")]).partial(format_instructions=parser.get_format_instructions())
    chain=prompt|invoke_prompt_value|parser
    tickets=pd.read_csv(RAW/"tickets.csv"); rows=[]; first_failure={}; mock_llm.reset(); llm_client.RATE_LIMIT_RETRIES=0
    for t in tickets.itertuples():
        status="manual_review"; record=None; attempts=0; messages=[]
        try:
            attempts=1
            record=chain.invoke({"ticket":t.text})
            status="valid"
        except Exception as exc:
            bad_reply=getattr(exc,"llm_output",None) or ""
            try:
                parsed=json.loads(bad_reply)
                if parsed.get("category") not in LABELS+ ("other",): kind="invalid_category"
                elif not 1 <= int(parsed.get("urgency",0)) <= 5: kind="urgency_out_of_range"
                else: kind="valid_json_schema_error"
            except Exception:
                kind="unreadable_ticket" if t.text.strip()=="..." else "invalid_json_chatty"
            first_failure[kind]=first_failure.get(kind,0)+1
            prompt_value=prompt.invoke({"ticket":t.text})
            messages=[{"role":"system","content":prompt_value.to_messages()[0].content},
                      {"role":"user","content":t.text},
                      {"role":"assistant","content":bad_reply},
                      {"role":"user","content":"VALIDATION ERROR: "+str(exc)}]
            for retry in range(2):
                attempts+=1
                reply=call_llm(messages)
                try:
                    record=parser.parse(reply); status="valid"; break
                except Exception as retry_exc:
                    if retry==1: break
                    messages.extend([{"role":"assistant","content":reply},
                                     {"role":"user","content":"VALIDATION ERROR: "+str(retry_exc)}])
        rows.append({"ticket_id":t.ticket_id,"status":status,"attempts":attempts,
                     "record":record.model_dump() if record is not None else None})
    with (OUT/"ticket_extractions.jsonl").open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row)+"\n")
    statuses=pd.Series([x["status"] for x in rows]).value_counts().to_dict()
    valid_records=[x["record"] for x in rows if x["record"] is not None]
    return {"status_counts":statuses,"first_attempt_failures":first_failure,
      "rate_limits_recovered":llm_client.RATE_LIMIT_RETRIES,"valid_records":len(valid_records),
      "wants_refund_true":sum(bool(x["wants_refund"]) for x in valid_records),
      "null_subscriber_id":sum(x["subscriber_id"] is None for x in valid_records)}

def chunk_text(text,size=400,overlap=60):
    """Split text at size-overlap strides; retain only starts with new content beyond overlap."""
    stride=size-overlap
    return [text[start:start+size] for start in range(0,len(text),stride) if start+overlap<len(text)]

def build_collection(size=400,overlap=60,collection_name="policies"):
    """Build an in-memory cosine Chroma collection from the five policy files."""
    client=chromadb.Client()
    try: client.delete_collection(collection_name)
    except Exception: pass
    ef=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    col=client.create_collection(collection_name,metadata={"hnsw:space":"cosine"},embedding_function=ef)
    ids=[]; docs=[]; metas=[]
    for file in sorted(POL.glob("*.txt")):
        for i,ch in enumerate(chunk_text(file.read_text(encoding="utf-8"),size,overlap)):
            ids.append(f"{file.stem}_{i}"); docs.append(ch); metas.append({"source":file.name,"chunk":i})
    col.add(ids=ids,documents=docs,metadatas=metas)
    return client,col,len(ids)

def retrieve(col,query,k=3):
    """Retrieve top-k policy chunks with cosine similarity (one minus Chroma distance)."""
    result=col.query(query_texts=[query],n_results=k)
    return [{"source":m["source"],"similarity":float(1-d),"text":doc}
      for doc,m,d in zip(result["documents"][0],result["metadatas"][0],result["distances"][0])]

def keyword_baseline(question):
    """Choose the policy file sharing the most non-stopword lowercase tokens with a question."""
    words=set(re.findall(r"[a-z]+",question.lower()))-mock_llm.STOP
    counts={}
    for file in sorted(POL.glob("*.txt")):
        content=file.read_text(encoding="utf-8").lower()
        file_words=set(re.findall(r"[a-z]+",content))-mock_llm.STOP
        counts[file.name]=len(words&file_words)
    return sorted(counts,key=lambda name:(-counts[name],name))[0]

POLICY_COLLECTION=None
POLICY_THRESHOLD=None

def answer(question):
    """Answer a policy question from top-three retrieved chunks or return the exact fallback."""
    if POLICY_COLLECTION is None or POLICY_THRESHOLD is None:
        raise RuntimeError("Build and calibrate the policy collection before calling answer().")
    hits=retrieve(POLICY_COLLECTION,question,k=3)
    if not hits or hits[0]["similarity"]<POLICY_THRESHOLD:
        return FALLBACK,[h["source"] for h in hits]
    context="\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    response=call_llm([{"role":"system","content":"POLICY ASSISTANT Answer using only the context below. If unsupported, say you do not have enough information."},
      {"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{question}"}])
    return response,[h["source"] for h in hits]

def run_rag():
    """Evaluate semantic retrieval, grounded answers, keyword baseline and chunk-size sensitivity."""
    client,col,count=build_collection()
    ev=pd.read_csv(RAW/"policy_eval.csv").fillna(""); rows=[]; sims_answer=[]; sims_unanswer=[]
    keyword_correct=0; top1_correct=0; top3_correct=0
    mock_llm.reset(); llm_client.RATE_LIMIT_RETRIES=0
    for q in ev.itertuples():
        hits=retrieve(col,q.question,3); top=hits[0]
        answerable=bool(q.expected_file)
        if answerable:
            sims_answer.append(top["similarity"])
            top1_correct+=int(top["source"]==q.expected_file)
            top3_correct+=int(any(h["source"]==q.expected_file for h in hits))
            keyword_correct+=int(keyword_baseline(q.question)==q.expected_file)
        else: sims_unanswer.append(top["similarity"])
        context="\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        # Threshold is computed after retrieving all 16 questions.
        rows.append({"question":q.question,"expected_file":q.expected_file,"top1_source":top["source"],
          "top1_similarity":top["similarity"],"hit_at_3":int(bool(q.expected_file) and any(h["source"]==q.expected_file for h in hits)),
          "context":context,"hits":hits,"expected_fact":q.expected_fact})
    threshold=(min(sims_answer)+max(sims_unanswer))/2
    global POLICY_COLLECTION,POLICY_THRESHOLD
    POLICY_COLLECTION=col; POLICY_THRESHOLD=threshold
    grounded_hits=0; fallback_count=0; ungrounded_answers=[]
    for row in rows:
        grounded,_sources=answer(row["question"])
        ungrounded=call_llm([{"role":"system","content":"POLICY ASSISTANT. Answer the question."},{"role":"user","content":row["question"]}])
        row["grounded_answer"]=grounded; row["ungrounded_answer"]=ungrounded
        if row["expected_fact"] and row["expected_fact"].lower() in grounded.lower(): grounded_hits+=1
        if not row["expected_file"] and grounded==FALLBACK: fallback_count+=1
        ungrounded_answers.append(ungrounded)
    output=pd.DataFrame([{k:r[k] for k in ["question","expected_file","top1_source","top1_similarity","hit_at_3","grounded_answer","ungrounded_answer"]} for r in rows])
    output.to_csv(OUT/"rag_eval.csv",index=False)

    # Smaller chunks: compare retrieval fact coverage and actual grounded mock answer coverage.
    _,small,small_count=build_collection(200,30,"policies_small")
    small_top3=0; small_answer_hits=0
    for q in ev[ev.expected_file.fillna("")!=""].itertuples():
        hits=retrieve(small,q.question,3)
        ctx="\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        if q.expected_fact.lower() in ctx.lower(): small_top3+=1
        ans=call_llm([{"role":"system","content":"POLICY ASSISTANT Answer using only the context below. If unsupported, say you do not have enough information."},
          {"role":"user","content":f"CONTEXT:\n{ctx}\nQUESTION:\n{q.question}"}])
        if q.expected_fact.lower() in ans.lower(): small_answer_hits+=1
    metrics={"chunk_count_400_60":count,"answerable_count":len(sims_answer),"unanswerable_count":len(sims_unanswer),
      "top3_file_hits":top3_correct,"top1_file_hits":top1_correct,"answerable_top1_similarity":{"min":min(sims_answer),"max":max(sims_answer),"mean":float(np.mean(sims_answer))},
      "unanswerable_top1_similarities":sims_unanswer,"keyword_baseline_top1_correct":keyword_correct,
      "rejection_threshold":threshold,"answerable_unanswerable_similarity_gap":min(sims_answer)-max(sims_unanswer),
      "grounded_expected_facts":grounded_hits,"unanswerable_fallbacks":fallback_count,
      "ungrounded_unique_answers":len(set(ungrounded_answers)),"small_chunk_count":small_count,
      "small_top3_fact_hits":small_top3,"small_grounded_fact_hits":small_answer_hits,
      "rate_limits_recovered":llm_client.RATE_LIMIT_RETRIES,
      "closest_unanswerable_question":rows[int(np.argmax([r["top1_similarity"] if not r["expected_file"] else -1 for r in rows]))]["question"],
      "closest_unanswerable_similarity":max(sims_unanswer)}
    return metrics

def run_client_demos():
    """Run the four required client demonstrations against the deterministic mock."""
    payload={"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"hello"}],"temperature":0.0,"max_tokens":300}
    mock_llm.reset()
    unauthorized=mock_llm.post(mock_llm.URL,headers={},json=payload).status_code
    mock_llm.reset()
    not_found=mock_llm.post("https://wrong.example/chat",headers={"Authorization":"Bearer demo"},json=payload).status_code
    mock_llm.reset()
    short_payload={**payload,"max_tokens":3}
    short=mock_llm.post(mock_llm.URL,headers={"Authorization":"Bearer demo"},json=short_payload).json()["choices"][0]["finish_reason"]
    mock_llm.reset()
    no_memory=call_llm([{"role":"user","content":"What is my name?"}])
    mock_llm.reset()
    with_history=call_llm([{"role":"user","content":"My name is Asha."},{"role":"assistant","content":"Nice to meet you."},{"role":"user","content":"What is my name?"}])
    mock_llm.reset(); llm_client.RATE_LIMIT_RETRIES=0
    for _ in range(30): call_llm([{"role":"user","content":"Say hello."}])
    return {"unauthorized_status":unauthorized,"wrong_url_status":not_found,"max_tokens_3_finish_reason":short,
      "no_memory_answer":no_memory,"same_request_with_history_answer":with_history,
      "thirty_calls_rate_limits_recovered":llm_client.RATE_LIMIT_RETRIES,"physical_post_calls":mock_llm._state["calls"]}

def run():
    """Run all Part D workflows and write machine-readable metrics."""
    OUT.mkdir(exist_ok=True)
    smoke=pydantic_smoke_tests()
    classifier,wrong=run_classifier()
    extraction=run_extraction()
    rag=run_rag()
    metrics={"client_demos":run_client_demos(),"pydantic_smoke_tests":smoke,"classifier":classifier,"few_shot_wrong_tickets":wrong,
             "extraction":extraction,"rag":rag}
    (OUT/"part_d_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": run()
