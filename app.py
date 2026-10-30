"""Production ASGI assembly: FastAPI + original V5 Gradio + Flask API."""
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.wsgi import WSGIMiddleware
from runtime import engine as nll
from api.flask_api import flask_app

app = FastAPI(title="NLL Universal Professional Editor")

@app.get("/health")
def health():
    return {"status": "ok", "parser_ready": "parser" in nll.nlp().pipe_names, "rules": len(nll.RULES), "must_paraphrase": bool(getattr(nll, "MUST_PARAPHRASE_ACTIVE", False))}

app.mount("/api", WSGIMiddleware(flask_app))
app = gr.mount_gradio_app(app, nll.NLL_APP, path="/")
