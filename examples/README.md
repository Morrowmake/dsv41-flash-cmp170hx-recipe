# Examples

Client examples for the running server: model `dsv41-flash`, base URL
`http://127.0.0.1:8000/v1`, no API key unless you set `API_KEY` in `.env`
(then export `API_KEY` before running these). Set `BASE_URL` to reach a
server on another machine ([Serving other machines](../docs/how-to-use.md#serving-other-machines)).

| File | What it shows |
|---|---|
| [`chat.sh`](chat.sh) | one chat request with `curl` and `jq` |
| [`chat.py`](chat.py) | the same with the OpenAI Python client, reasoning and answer separately |
| [`stream.py`](stream.py) | streaming, with reasoning and answer tokens told apart |
| [`tool_call.py`](tool_call.py) | a full tool-call round trip |
| [`tp4.env`](tp4.env) | a starting `.env` for serving other machines with a key |

The Python examples need `pip install openai`. The `.env` file is a starting
point for the repository's `.env`; every setting is described in
[Settings](../docs/how-to-use.md#settings).
