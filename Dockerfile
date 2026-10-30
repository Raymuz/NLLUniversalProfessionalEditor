FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860 NLL_ENABLE_LANGUAGETOOL=false
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader wordnet omw-1.4
COPY . .
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 CMD curl -fsS http://127.0.0.1:7860/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--proxy-headers"]
