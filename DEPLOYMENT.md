# Hugging Face deployment

Target: `raymuz/NLLUniversalProfessionalEditor` (existing Space). This package **does not automatically overwrite** the current Space.

## Preflight

1. Preserve the existing Space repository/revision before uploading.
2. Run `python -m compileall -q app.py api runtime tests` and, with installed dependencies and the English spaCy model, `python -m tests.test_smoke`.
3. Start locally with `uvicorn app:app --host 0.0.0.0 --port 7860`, then test `/health`, `/api/health`, `/api/rewrite`, and the UI.
4. Use Docker CPU basic first; GPU not required for deterministic rules.

## Publish after tests pass

Upload package contents at the repository root (not its enclosing folder). The `README.md` declares `sdk: docker` and port 7860. Docker will install packages and the spaCy model, then run uvicorn. This changes the existing Space configuration to Docker; verify the current Space SDK and preserve a rollback before the commit.

For Colab publishing, authenticate through `huggingface_hub` interactively or via Colab secrets and call `HfApi().upload_folder(folder_path=..., repo_id="raymuz/NLLUniversalProfessionalEditor", repo_type="space")`. Do not hardcode the token.

## Verification

- Build logs show successful package install and spaCy model installation
- `GET /health` reports parser ready and 30 rules
- `GET /api/health` returns success
- UI loads and performs editing and downloads
- `POST /api/rewrite` returns a real `DocumentResult`
- Must-paraphrase failure retains input and reason

If anything fails, restore the preserved revision.

No live HF deployment was executed by package generation.
