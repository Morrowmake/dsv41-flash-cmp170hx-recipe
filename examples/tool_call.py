"""A tool call round trip with the OpenAI Python client (pip install openai).

The model asks for get_weather; we answer with a made-up result and let it
write the final reply. The same tool as `./start.sh smoke`.
"""
import json
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.environ.get("API_KEY", "none"),
)
MODEL = "dsv41-flash"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

# The model reasons before it calls the tool: leave room for both.
messages = [{"role": "user", "content": "What is the weather in Reykjavik right now? Use the tool."}]
first = client.chat.completions.create(
    model=MODEL, messages=messages, tools=tools, tool_choice="auto", max_tokens=4096
)
msg = first.choices[0].message
if not msg.tool_calls:
    print("no tool call:", msg.content)
    raise SystemExit(1)

call = msg.tool_calls[0]
args = json.loads(call.function.arguments)
print(f"tool_call: {call.function.name}({args})")

# Stand-in for a real weather lookup.
result = {"city": args["city"], "temperature": 7, "unit": args.get("unit", "celsius"), "sky": "overcast"}

messages.append(msg.model_dump(exclude_none=True))
messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, max_tokens=4096)
print("answer:", final.choices[0].message.content)
