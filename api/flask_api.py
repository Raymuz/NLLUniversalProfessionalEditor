"""HTTP API using the original runtime ENGINE. No sample or fake results."""
from flask import Flask, jsonify, request
from dataclasses import asdict, is_dataclass
from runtime import engine as nll

flask_app = Flask(__name__)
flask_app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024

@flask_app.get("/health")
def health():
    return jsonify(status="ok", rule_count=len(nll.RULES), parser_ready="parser" in nll.nlp().pipe_names)

@flask_app.post("/rewrite")
def rewrite():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="text must be a nonempty string"), 400
    if len(text) > 50000:
        return jsonify(error="text exceeds 50,000 characters"), 413
    workflow = data.get("workflow", "Edit / Paraphrase Only")
    mode = data.get("mode", "Balanced")
    discipline = data.get("discipline", "Auto Detect")
    terms = data.get("protected_terms", "")
    if workflow not in ("Scan Only", "Scan & Neutralize", "Edit / Paraphrase Only"):
        return jsonify(error="unsupported workflow"), 400
    if mode not in ("Minimal", "Balanced", "Structural"):
        return jsonify(error="unsupported mode"), 400
    if not isinstance(terms, str):
        return jsonify(error="protected_terms must be string"), 400
    try:
        try:
            r = nll.ENGINE.process(text, workflow, mode, discipline, terms, None)
        except TypeError:
            r = nll.ENGINE.process(text=text, workflow=workflow, mode=mode, discipline=discipline, terms_text=terms, progress=None)
        value = asdict(r) if is_dataclass(r) else r
        return jsonify(value)
    except Exception:
        flask_app.logger.exception("NLL request failed")
        return jsonify(error="processing failed; inspect server logs"), 500
