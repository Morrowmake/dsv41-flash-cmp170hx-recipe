"""One chat request with the OpenAI Python client (pip install openai).

The server listens on 127.0.0.1:8000 by default and has no API key; the client
still needs a value, so any string works unless you set API_KEY in .env.
"""
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.environ.get("API_KEY", "none"),
)

resp = client.chat.completions.create(
    model="dsv41-flash",
    messages=[{"role": "user", "content": "Write a haiku about PCIe."}],
    max_tokens=2048,
    # How much the model reasons: "low", "high" (the default), "xhigh" or "max".
    # To turn the reasoning off: extra_body={"chat_template_kwargs": {"thinking": False}}
    reasoning_effort="low",
)
msg = resp.choices[0].message
# The reasoning comes back in its own field, the answer in `content`.
print("reasoning:", getattr(msg, "reasoning", None) or "")
print("answer:", msg.content)
