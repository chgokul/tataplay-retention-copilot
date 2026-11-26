"""Insert measured Part D and retention-copilot results into README.md."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"; README=ROOT/"README.md"

def table(headers, rows):
    """Render a simple Markdown table from headers and row values."""
    return "| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+"".join("| "+" | ".join(map(str,row))+" |\n" for row in rows)

def run():
    """Read Part D JSON metrics and replace the README results marker block."""
    d=json.loads((OUT/"part_d_metrics.json").read_text(encoding="utf-8"))
    c=json.loads((OUT/"copilot_metrics.json").read_text(encoding="utf-8"))
    lines=[]
    demo=d["client_demos"]
    lines.append("### Client demonstrations\n\n")
    lines.append(table(["Check","Observed result"],[
      ("Missing Authorization",demo["unauthorized_status"]),("Wrong URL",demo["wrong_url_status"]),
      ("max_tokens=3",demo["max_tokens_3_finish_reason"]),("No history",demo["no_memory_answer"]),
      ("Same request with history",demo["same_request_with_history_answer"]),
      ("30 calls: recovered rate limits / physical posts",f"{demo['thirty_calls_rate_limits_recovered']} / {demo['physical_post_calls']}")]))
    smoke=d["pydantic_smoke_tests"]
    lines.append("\nPydantic coerces urgency string `3` to integer **3**. `urgency=9` and `category='refund'` raise field validation errors: `"+json.dumps(smoke["urgency_9"])+"` and `"+json.dumps(smoke["category_refund"])+"`.\n\n")

    lines.append("### Ticket classification\n\n")
    cls=d["classifier"]
    lines.append(table(["Prompt","Valid labels","Correct","Accuracy","Recovered 429s"],[(k,v["valid_labels"],v["correct"],f"{v['accuracy']*100:.2f}%",v["recovered_rate_limits"]) for k,v in cls.items()]))
    lines.append("\nFew-shot errors (the readable ticket IDs and predicted/true labels) are: "+", ".join(f"{x['ticket_id']} ({x['few_shot']} vs {x['true_category']})" for x in d["few_shot_wrong_tickets"])+". The remaining errors contain multiple intents; a production classifier should identify the primary requested action first or represent multiple intents explicitly. Chain-of-thought is justified only when multi-step reasoning must be audited; these short fixed-label tickets are better handled with concise prompts and examples.\n\n")

    ext=d["extraction"]
    lines.append("### Validated ticket extraction\n\n")
    lines.append(table(["Measure","Result"],[("Outcome counts",json.dumps(ext["status_counts"])),
      ("First-attempt failure categories",json.dumps(ext["first_attempt_failures"])),
      ("Valid records",ext["valid_records"]),("wants_refund = true",ext["wants_refund_true"]),
      ("Null subscriber_id",ext["null_subscriber_id"]),("Recovered rate limits",ext["rate_limits_recovered"])]))
    lines.append("\nTickets still marked `manual_review` are routed to a human support queue with the original text and validation errors; they are not silently discarded or answered with an invented record.\n\n")

    rag=d["rag"]
    lines.append("### Policy retrieval and grounded answering\n\n")
    lines.append(table(["Retrieval measure","Result"],[("400/60 chunk count",rag["chunk_count_400_60"]),
      ("Answerable top-3 file hits",f"{rag['top3_file_hits']} / 12"),("Answerable top-1 file hits",f"{rag['top1_file_hits']} / 12"),
      ("Answerable top-1 similarity min–max–mean",f"{rag['answerable_top1_similarity']['min']:.3f} – {rag['answerable_top1_similarity']['max']:.3f} – {rag['answerable_top1_similarity']['mean']:.3f}"),
      ("Unanswerable top-1 similarities",", ".join(f"{x:.3f}" for x in rag["unanswerable_top1_similarities"])),
      ("Keyword baseline top-1 correct",f"{rag['keyword_baseline_top1_correct']} / 12"),
      ("Rejection threshold",f"{rag['rejection_threshold']:.3f}"),("Similarity gap between ranges",f"{rag['answerable_unanswerable_similarity_gap']:.3f}"),
      ("Grounded answers containing expected fact",f"{rag['grounded_expected_facts']} / 12"),
      ("Unanswerable questions receiving fallback",f"{rag['unanswerable_fallbacks']} / 4"),
      ("Unique ungrounded answers",rag["ungrounded_unique_answers"]),
      ("200/30 chunk count",rag["small_chunk_count"]),("200/30 top-3 fact coverage",f"{rag['small_top3_fact_hits']} / 12"),
      ("200/30 grounded expected facts",f"{rag['small_grounded_fact_hits']} / 12")]))
    lines.append("\nMeaning-based retrieval helps because it can match paraphrases and related concepts even when the question and policy use different words; keyword overlap misses such matches and can be distracted by generic shared terms. The unanswerable question closest to the rejection boundary was **"+rag["closest_unanswerable_question"]+f"** (top-1 similarity {rag['closest_unanswerable_similarity']:.3f}). The observed gap between the weakest answerable and strongest unanswerable question is only {rag['answerable_unanswerable_similarity_gap']:.3f}, so the threshold has limited safety margin and needs more labelled calibration examples. Smaller chunks can split a policy sentence or separate a condition from its action, reducing the chance that all evidence appears together in the top three results.\n\n")

    lines.append("### Retention briefs\n\n")
    lines.append(table(["Measure","Result"],[("Valid and consistent briefs",c["valid_briefs"]),
      ("Risk-band counts",json.dumps(c["risk_band_counts"])),("Recommended-action counts",json.dumps(c["recommended_action_counts"]))]))
    lines.append("\nAll 15 briefs are validated against the Pydantic schema, checked against the source subscriber and risk band, and required to cite `retention_offers_policy.txt`.\n")

    text=README.read_text(encoding="utf-8")
    start="<!-- PART_D_RESULTS_START -->"; end="<!-- PART_D_RESULTS_END -->"
    a=text.index(start)+len(start); b=text.index(end,a)
    README.write_text(text[:a]+"\n\n"+"".join(lines)+"\n"+text[b:],encoding="utf-8")
    print("README Part D results updated from outputs/part_d_metrics.json and outputs/copilot_metrics.json")

if __name__=="__main__": run()
