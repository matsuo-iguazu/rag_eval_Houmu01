#!/usr/bin/env python3
# evaluator.py -- OpenAI v1 (openai>=1.0.0) 対応・BOM対応・明示評価ルール

import os
import csv
import json
import re
import time
from openai import OpenAI

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise SystemExit("Set OPENAI_API_KEY in environment")

client = OpenAI(api_key=OPENAI_KEY)

MODEL = "gpt-4o-mini"  # 必要に応じて変更
INPUT_CSV = "data/rag_eval_input.csv"
OUTPUT_CSV = "data/rag_eval_output.csv"

SYSTEM_PROMPT = """あなたは与えられた「質問文」「参考文章」「RAG出力結果」を比較し、RAG出力結果の正確性を評価します。
評価ルール:
- 5: 質問に正確に回答しており、かつ参考文章以上の情報や背景も正しく付加されている（上級者レベル）
- 4: 質問に正確に回答しており、参考文章に照らしても正しい（一般レベル）
- 3: 質問に概ね正しく回答しているが、一部情報が不足・誤解の可能性あり
- 2: 質問に対する判断が誤っている
- 1: 回答されていない
- 0: 質問自体が不適切（参考文章に答えがない）
出力はJSONで返してください（例: {"score": <0-5>, "reason": "<簡潔な理由>"}）。
"""

def normalize_header(name):
    return (name or "").strip().lower()

def build_map(fieldnames):
    norm = {normalize_header(f): f for f in fieldnames}
    keys = ["question", "toc", "reference", "rag_output", "quote"]
    return {k: norm.get(k) for k in keys}

def _extract_content_from_choice(choice):
    try:
        if hasattr(choice, "message") and choice.message is not None:
            msg = choice.message
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
            if hasattr(msg, "content"):
                return getattr(msg, "content") or ""
            try:
                return msg.get("content", "") or ""
            except Exception:
                return ""
    except Exception:
        pass
    try:
        if isinstance(choice, dict):
            return (choice.get("message", {}) or {}).get("content", "") or choice.get("text", "") or ""
    except Exception:
        pass
    try:
        if hasattr(choice, "text"):
            return getattr(choice, "text") or ""
    except Exception:
        pass
    return ""

def call_model(question: str, reference: str, rag_output: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"質問文:\n{question}\n\n参考文章:\n{reference}\n\nRAG出力結果:\n{rag_output}"}
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as e:
        return {"score": -1, "reason": f"API error: {e}"}

    choices = []
    try:
        if hasattr(resp, "choices"):
            choices = list(resp.choices) or []
        elif isinstance(resp, dict):
            choices = resp.get("choices", []) or []
    except Exception:
        try:
            choices = resp.get("choices", []) if isinstance(resp, dict) else []
        except Exception:
            choices = []

    if not choices:
        return {"score": -1, "reason": f"empty response or no choices: {repr(resp)[:400]}"}

    choice = choices[0]
    content = _extract_content_from_choice(choice)
    content = (content or "").strip()
    if not content:
        return {"score": -1, "reason": f"empty model response content, raw: {repr(resp)[:400]}"}

    try:
        j = json.loads(content)
        score = int(j.get("score"))
        reason = str(j.get("reason", "")).strip()
        return {"score": score, "reason": reason}
    except Exception:
        m = re.search(r"\b([0-5])\b", content)
        if m:
            return {"score": int(m.group(1)), "reason": content[:500]}
        return {"score": -1, "reason": f"unparsed model output: {content[:500]}"}

def main():
    with open(INPUT_CSV, encoding="utf-8-sig", newline='') as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise SystemExit("Input CSV missing header or empty")
        hdr_map = build_map(reader.fieldnames)
        if not hdr_map["question"] or not hdr_map["reference"] or not hdr_map["rag_output"]:
            raise SystemExit(f"CSV must contain headers: question,toc,reference,rag_output,quote. Detected: {reader.fieldnames}")

        out_fields = list(reader.fieldnames) + ["score", "reason"]
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for i, row in enumerate(reader, start=1):
                q = (row.get(hdr_map["question"]) or "").strip()
                ref = (row.get(hdr_map["reference"]) or "").strip()
                out = (row.get(hdr_map["rag_output"]) or "").strip()
                if not (q or ref or out):
                    continue
                res = call_model(q, ref, out)
                row["score"] = res.get("score")
                row["reason"] = res.get("reason", "")
                writer.writerow(row)
                time.sleep(0.3)

    print("Done:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
