"""Offline-first chat client with bounded rate-limit retries."""
import os, time
import mock_llm

BASE_DELAY=0.05
MODEL="openai/gpt-4o-mini"

def call_llm(messages, temperature=0.0, max_tokens=300):
    """Call the offline mock or optional live endpoint, retrying HTTP 429 only."""
    payload={"model":MODEL,"messages":messages,"temperature":temperature,"max_tokens":max_tokens}
    if os.getenv("LLM_MODE","mock").lower()=="live":
        import requests
        url=os.getenv("LLM_URL","https://openrouter.ai/api/v1/chat/completions")
        key=os.getenv("OPENROUTER_API_KEY","")
        if not key: raise RuntimeError("OPENROUTER_API_KEY is required in live mode")
        post=lambda: requests.post(url,headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json=payload,timeout=30)
    else:
        url=mock_llm.URL
        post=lambda: mock_llm.post(url,headers={"Authorization":"Bearer offline-token","Content-Type":"application/json"},json=payload)
    for retry in range(4):
        response=post()
        if response.status_code==429 and retry<3:
            time.sleep(BASE_DELAY*(2**retry)); continue
        if response.status_code!=200:
            raise RuntimeError(f"LLM request failed: HTTP {response.status_code}: {response.text}")
        body=response.json()
        choice=body["choices"][0]
        if choice.get("finish_reason")=="length":
            print("WARNING: model response was truncated (finish_reason=length)")
        return choice["message"]["content"]
    raise RuntimeError("LLM request failed after 3 rate-limit retries")
