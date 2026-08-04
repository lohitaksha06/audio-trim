"""Optional LLM-powered prompt parsing.

`llm_plan_from_prompt` sends the user's natural-language request to an LLM
(OpenAI-compatible API, or a local transformers model) and asks it to return
a structured JSON plan matching our `PromptPlan` schema.

Configuration via environment variables:
  LLM_API_KEY      — API key for the OpenAI-compatible endpoint.
  LLM_BASE_URL     — e.g. "https://api.openai.com/v1".
  LLM_MODEL        — model id, e.g. "gpt-4o-mini".
  LLM_LOCAL_MODEL  — optional HF model id (e.g. "Qwen/Qwen2.5-0.5B-Instruct")
                     to run inference locally via transformers when no API key
                     is present.

If no provider is configured (or the call fails), returns ``None`` so the
caller can fall back to the deterministic regex engine.
"""

import json
import os
import re
from typing import Any

from server.ml.prompt_engine import Intent, PromptPlan

SYSTEM_PROMPT = (
    "You convert a single audio-editing request into JSON. "
    "Return ONLY JSON, no prose. Schema: "
    '{"intent": <string>, "params": {<stringified values>}}. '
    "intent must be one of: " + ", ".join(i.value for i in Intent) + ". "
    "Follow these rules:\n"
    "- trim -> params:{start, end} in seconds\n"
    "- remove -> params:{start, end} or {instrument} (vocals/drums/bass/guitar/keys/other)\n"
    "- paint -> params:{start, end} (seamlessly fill over a range)\n"
    "- separate -> params:{}\n- convert -> params:{format}\n"
    "- isolate -> params:{instrument}\n"
    "- fade -> params:{fade_in, fade_out}\n"
    "- normalize -> params:{}\n"
    "- remove_silence -> params:{}\n"
    "- remove_fillers -> params:{}\n"
    "- mood -> params:{mood} in {dark, bright, energetic, calm}\n"
    "- speed -> params:{speed_factor}\n"
    "- reverb -> params:{reverb_amount}\n"
    "- unknown -> params:{}\n"
    "Convert all timestamps ('2:30', '90s', 'at 0:45') to seconds floats. "
    "Only include params that apply."
)


def _sanitize_params(params: dict) -> dict:
    return {k: v for k, v in params.items() if v is not None}


def _parse_json_reply(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _local_predict(messages: list[dict]) -> str | None:
    model_id = os.environ.get("LLM_LOCAL_MODEL")
    if not model_id:
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        prompt = SYSTEM_PROMPT + "\n\nRequest: " + messages[-1]["content"]
        inp = tok(prompt, return_tensors="pt")
        out = model.generate(**inp, max_new_tokens=120, do_sample=False)
        return tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
    except Exception:
        return None


def _api_prompt(messages: list[dict], model: str) -> str | None:
    import requests

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return None
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "temperature": 0.0},
        timeout=30,
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def llm_plan_from_prompt(prompt: str) -> PromptPlan | None:
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = _api_prompt(messages, model) or _local_predict(messages)
    if not text:
        return None

    parsed = _parse_json_reply(text)
    if not parsed or "intent" not in parsed:
        return None

    intent = parsed["intent"]
    try:
        intent_enum = Intent(intent)
    except ValueError:
        intent_enum = Intent.UNKNOWN

    params = parsed.get("params") or {}
    return PromptPlan(intent=intent_enum, params=params, raw_prompt=prompt)