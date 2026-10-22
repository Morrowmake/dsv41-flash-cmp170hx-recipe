"""Streaming chat with the OpenAI Python client (pip install openai).

Reasoning tokens and answer tokens arrive in separate delta fields; this prints
the reasoning dimmed and the answer as normal text.
"""
import os
import sys

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.environ.get("API_KEY", "none"),
)

stream = client.chat.completions.create(
    model="dsv41-flash",
    messages=[{"role": "user", "content": "Explain tensor parallelism in three sentences."}],
    max_tokens=4096,
    stream=True,
)
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
    if reasoning:
        sys.stdout.write(f"\033[2m{reasoning}\033[0m")
    if delta.content:
        sys.stdout.write(delta.content)
    sys.stdout.flush()
print()
