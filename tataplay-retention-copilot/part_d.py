
"""Part D: ticket workflows and policy RAG.
Requires the Part D dependencies.
"""

from pathlib import Path
import json

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

from pydantic import BaseModel, Field
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser
)

from llm_client import call_llm
import mock_llm


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
POL = ROOT / "data" / "policies"
OUT = ROOT / "outputs"

LABELS = ("billing", "signal", "plan_change", "cancellation")


# --------------------------------------------------
# 1. PYDANTIC MODELS
# --------------------------------------------------

class TicketRecord(BaseModel):
    """Validated structured representation of a support ticket."""

    subscriber_id: Optional[str] = None

    category: Literal[
        "billing",
        "signal",
        "plan_change",
        "cancellation",
        "other"
    ]

    urgency: int = Field(ge=1, le=5)
    wants_refund: bool


class RetentionBrief(BaseModel):
    """Validated retention action and policy reference."""

    subscriber_id: str
    risk_band: Literal["High", "Medium", "Low"]
    recommended_action: str
    policy_source: str


# --------------------------------------------------
# 2. CLASSIFICATION HELPER
# --------------------------------------------------

def normalize_label(value):
    """Normalize classifier output for label comparison."""

    return str(value).strip().lower().strip("`'\" ")


# --------------------------------------------------
# 3. TICKET CLASSIFICATION
# --------------------------------------------------

def run_classifier():

    print("\n===== RUNNING CLASSIFIER =====")

    tickets = pd.read_csv(RAW / "tickets.csv")

    tickets = tickets[tickets["text"] != "..."].copy()

    examples = pd.read_csv(RAW / "few_shot_examples.csv")

    strong = (
        "TICKET CLASSIFIER. "
        "You are a support ticket classifier. "
        "Classify each ticket into exactly one of "
        "billing, signal, plan_change, cancellation. "
        "Respond with only the label. "
        "Do not create new labels."
    )

    few = strong + "\n" + "\n".join(
        f"Ticket: {r.text}\nLabel: {r.label}"
        for r in examples.itertuples()
    )

    prompts = [
        ("weak", "TICKET CLASSIFIER. Classify the ticket."),
        ("zero_shot", strong),
        ("few_shot", few)
    ]

    outputs = {}

    for name, system in prompts:

        mock_llm.reset()

        vals = []

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{ticket}")
        ])

        def invoke(pv):

            msgs = [
                {
                    "role": "system" if m.type == "system" else "user",
                    "content": m.content
                }
                for m in pv.to_messages()
            ]

            return call_llm(msgs)

        chain = prompt | invoke | StrOutputParser()

        for t in tickets.itertuples():

            prediction = chain.invoke({
                "ticket": t.text
            })

            vals.append(prediction)

        # Preserve raw predictions for debugging.
        raw_predictions = vals

        # Normalize formatting differences.
        normalized_predictions = [
            normalize_label(x)
            for x in raw_predictions
        ]

        outputs[name] = normalized_predictions

        valid = sum(
            prediction in LABELS
            for prediction in normalized_predictions
        )

        invalid = len(normalized_predictions) - valid

        correct = sum(
            predicted == actual
            for predicted, actual in zip(
                normalized_predictions,
                tickets["true_category"]
            )
        )

        accuracy = (
            correct / len(tickets)
            if len(tickets) > 0
            else 0
        )

        print(
            "\nCLASSIFIER",
            name,
            "| Valid:", valid,
            "| Invalid:", invalid,
            "| Accuracy:", round(accuracy, 4)
        )

        # Detailed weak classifier debugging.
        if name == "weak":

            print("\n===== WEAK CLASSIFIER DEBUG =====")

            for actual, raw, predicted in zip(
                tickets["true_category"],
                raw_predictions,
                normalized_predictions
            ):

                print(
                    "Actual:", actual,
                    "| Raw prediction:", repr(raw),
                    "| Normalized:", repr(predicted)
                )

            print("===== END DEBUG =====\n")

    # Save predictions from all three classifiers.
    result = pd.DataFrame({
        "ticket_id": tickets["ticket_id"],
        "true_category": tickets["true_category"],
        "weak": outputs["weak"],
        "zero_shot": outputs["zero_shot"],
        "few_shot": outputs["few_shot"]
    })

    result.to_csv(
        OUT / "ticket_classification.csv",
        index=False
    )

    print("\nClassification results saved.")


# --------------------------------------------------
# 4. STRUCTURED EXTRACTION WITH PYDANTIC
# --------------------------------------------------

def run_extraction():

    print("\n===== RUNNING EXTRACTION =====")

    parser = PydanticOutputParser(
        pydantic_object=TicketRecord
    )

    system = (
        "TICKET EXTRACTOR. "
        "Extract the ticket fields. "
        + parser.get_format_instructions()
    )

    tickets = pd.read_csv(RAW / "tickets.csv")

    rows = []

    mock_llm.reset()

    for t in tickets.itertuples():

        messages = [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": t.text
            }
        ]

        status = "manual_review"
        record = None
        attempts = 0

        for attempt in range(3):

            attempts += 1

            reply = call_llm(messages)

            try:

                record = parser.parse(reply)

                status = "valid"

                break

            except Exception as exc:

                if attempt == 2:
                    break

                messages += [
                    {
                        "role": "assistant",
                        "content": reply
                    },
                    {
                        "role": "user",
                        "content": "VALIDATION ERROR: " + str(exc)
                    }
                ]

        rows.append({
            "ticket_id": t.ticket_id,
            "status": status,
            "attempts": attempts,
            "record": record.model_dump() if record else None
        })

    with (OUT / "ticket_extractions.jsonl").open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:

            f.write(json.dumps(row) + "\n")

    print(
        "EXTRACTION",
        pd.Series(
            [r["status"] for r in rows]
        ).value_counts().to_dict()
    )


# --------------------------------------------------
# 5. DOCUMENT CHUNKING
# --------------------------------------------------

def chunk_text(text, size=400, overlap=60):

    stride = size - overlap

    if stride <= 0:
        raise ValueError("Chunk size must be greater than overlap.")

    return [
        text[i:i + size]
        for i in range(0, len(text), stride)
    ]


# --------------------------------------------------
# 6. CHROMADB COLLECTION
# --------------------------------------------------

def build_collection(size=400, overlap=60):

    print("\n===== BUILDING POLICY VECTOR DATABASE =====")

    client = chromadb.Client()

    try:
        client.delete_collection("policies")
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    col = client.create_collection(
        "policies",
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef
    )

    ids = []
    docs = []
    metas = []

    for file in sorted(POL.glob("*.txt")):

        content = file.read_text(encoding="utf-8")

        chunks = chunk_text(
            content,
            size=size,
            overlap=overlap
        )

        for i, ch in enumerate(chunks):

            ids.append(f"{file.stem}_{i}")

            docs.append(ch)

            metas.append({
                "source": file.name,
                "chunk": i
            })

    if ids:

        col.add(
            ids=ids,
            documents=docs,
            metadatas=metas
        )

    print("Policy chunks indexed:", len(ids))

    return client, col


# --------------------------------------------------
# 7. SEMANTIC RETRIEVAL
# --------------------------------------------------

def retrieve(col, query, k=3):

    if col.count() == 0:
        return []

    result = col.query(
        query_texts=[query],
        n_results=min(k, col.count())
    )

    return [
        {
            "source": m["source"],
            "similarity": 1 - d,
            "text": doc
        }
        for doc, m, d in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0]
        )
    ]


# --------------------------------------------------
# 8. RAG EVALUATION
# --------------------------------------------------

def run_rag():

    print("\n===== RUNNING RAG EVALUATION =====")

    client, col = build_collection()

    ev = pd.read_csv(RAW / "policy_eval.csv")

    rows = []

    for q in ev.itertuples():

        hits = retrieve(
            col,
            q.question
        )

        context = "\n\n".join(
            f"[{h['source']}]\n{h['text']}"
            for h in hits
        )

        grounded = (
            "I don't have enough information to answer that."
        )

        if hits and hits[0]["similarity"] >= 0.32:

            grounded = call_llm([
                {
                    "role": "system",
                    "content": (
                        "POLICY ASSISTANT. "
                        "Answer using only the context below. "
                        "If unsupported, say you do not have "
                        "enough information."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXT:\n{context}\n"
                        f"QUESTION:\n{q.question}"
                    )
                }
            ])

        ungrounded = call_llm([
            {
                "role": "system",
                "content": "POLICY ASSISTANT. Answer the question."
            },
            {
                "role": "user",
                "content": q.question
            }
        ])

        rows.append({
            "question": q.question,
            "expected_file": q.expected_file,
            "top1_source": hits[0]["source"] if hits else "",
            "top1_similarity": (
                hits[0]["similarity"] if hits else 0
            ),
            "hit_at_3": int(
                any(
                    h["source"] == q.expected_file
                    for h in hits
                )
            ),
            "grounded_answer": grounded,
            "ungrounded_answer": ungrounded
        })

    pd.DataFrame(rows).to_csv(
        OUT / "rag_eval.csv",
        index=False
    )

    print("RAG evaluation saved.")

    return client, col


# --------------------------------------------------
# 9. MAIN PART D PIPELINE
# --------------------------------------------------

def run():

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    run_classifier()

    run_extraction()

    run_rag()

    print("\n===== PART D COMPLETED SUCCESSFULLY =====")


if __name__ == "__main__":

    run()
