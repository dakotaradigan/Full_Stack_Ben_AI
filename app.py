import json
import os
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

import chatbot

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "devsecret")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

with open("benchmarks.json", "r") as f:
    bench_data = json.load(f)["benchmarks"]

filter_options: Dict[str, set[str]] = {}
for bench in bench_data:
    tags = bench.get("tags", {})
    for k, v in tags.items():
        if isinstance(v, list):
            filter_options.setdefault(k, set()).update(v)
        else:
            filter_options.setdefault(k, set()).add(str(v))


@app.route("/")
def index():
    options = {k: sorted(list(v)) for k, v in filter_options.items()}
    return render_template("index.html", options=options)


def process_message(user_message: str) -> str:
    if "messages" not in session:
        session["messages"] = [
            {"role": "system", "content": chatbot.SYSTEM_PROMPT}
        ]
        session["resp_count"] = 0

    messages = session["messages"]
    resp_count = session["resp_count"]

    messages.append({"role": "user", "content": user_message})
    try:
        response = chatbot._with_retry(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=[{"type": "function", "function": func} for func in chatbot.FUNCTIONS],
            tool_choice="auto",
        )
    except Exception:
        app.logger.exception("Failed to create chat completion")
        return "Sorry, something went wrong. Please try again later."
    msg = response.choices[0].message
    final = ""
    if msg.tool_calls:
        messages.append({"role": "assistant", "content": None, "tool_calls": msg.tool_calls})
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = chatbot.call_function(call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })
        try:
            follow = chatbot._with_retry(
                model="gpt-3.5-turbo",
                messages=messages,
            )
        except Exception:
            app.logger.exception("Failed to generate follow-up response")
            return "Sorry, something went wrong. Please try again later."
        final = follow.choices[0].message.content
    else:
        final = msg.content or ""

    resp_count += 1
    if resp_count % chatbot.DISCLAIMER_FREQUENCY == 0:
        final = f"{final}\n\n{chatbot.DISCLAIMER_TEXT}"

    messages.append({"role": "assistant", "content": final})
    session["messages"] = messages
    session["resp_count"] = resp_count
    return final


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(force=True)
    user_message = data.get("message", "")
    reply = process_message(user_message)
    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(debug=True)
