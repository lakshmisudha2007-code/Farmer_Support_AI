import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from chatbot_config import SYSTEM_PROMPT

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

MODEL_NAME = "gemini-3.1-flash-lite"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Please enter a message."}), 400

    if client is None:
        return jsonify({
            "error": "Gemini API key is not configured. Please add GEMINI_API_KEY to the .env file."
        }), 500

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_message,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.4,
            },
        )

        answer = (response.text or "").strip()

        if not answer:
            answer = "Sorry, I could not generate a response. Please try again."

        return jsonify({"response": answer})

    except Exception as exc:
        print(f"CHAT ERROR: {exc}")
        return jsonify({
            "error": "Sorry, something went wrong while processing your request."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
