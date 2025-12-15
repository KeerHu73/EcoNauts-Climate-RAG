import os
import sys
from flask import Flask, request, jsonify

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rag_pipeline.rag import answer_question

app = Flask(__name__)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    # optional k, enforce rubric: k >= 3
    try:
        k = int(data.get("k", 3))
    except (TypeError, ValueError):
        k = 3
    if k < 3:
        k = 3

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    try:
        result = answer_question(question, k=k)
        return jsonify(result), 200
    except Exception as e:
        # keep it simple; you can log e if you want
        return jsonify({"error": "Internal error while answering the question."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



    