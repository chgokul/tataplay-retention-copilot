"""Offline-first chat-completion client with bounded rate-limit retries."""
import os
import time
import mock_llm

BASE_DELAY=0.05
MODEL="openai/gpt-4o-mini"
RATE_LIMIT_RETRIES=0

def call_llm(messages, temperature=0.0, max_tokens=300):
    """Return assistant text; retry HTTP 429 up to three times and fail on other statuses."""
    global RATE_LIMIT_RETRIES
    payload={"model":MODEL,"messages":messages,"temperature":temperature,"max_tokens":max_tokens}
    headers={"Content-Type":"application/json"}
    if os.getenv("LLM_MODE","mock").lower()=="live":
        import requests
        key=os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("LLM_MODE=live requires OPENROUTER_API_KEY in the environment")
        headers["Authorization"]=f"Bearer {key}"
        url=os.getenv("LLM_URL","https://openrouter.ai/api/v1/chat/completions")
        post=lambda: requests.post(url,headers=headers,json=payload,timeout=30)
    else:
        headers["Authorization"]="Bearer offline-token"
        url=mock_llm.URL
        post=lambda: mock_llm.post(url,headers=headers,json=payload)
    for retry in range(4):
        response=post()
        if response.status_code==429 and retry<3:
            RATE_LIMIT_RETRIES += 1
            time.sleep(BASE_DELAY*(2**retry))
            continue
        if response.status_code!=200:
            raise RuntimeError(f"LLM request failed with HTTP {response.status_code}: {response.text}")
        body=response.json(); choice=body["choices"][0]
        if choice.get("finish_reason")=="length":
            print("WARNING: LLM response was truncated (finish_reason=length)")
        return choice["message"]["content"]
    raise RuntimeError("LLM request failed after 3 rate-limit retries")
