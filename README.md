---
title: NLL Universal Professional Editor
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Non-generative academic editing and review
---

# NLL Universal Professional Editor

This release preserves the user's uploaded V5 30-rule deterministic transformation engine, its safety validator, must-paraphrase routing and existing Gradio UI. Candidate changes are rejected when safeguards fail; unchanged text is not proof that no alternative could exist. Writing-pattern findings do not identify authorship.

Health: `/health`; Flask API: `/api/health` and `/api/rewrite`. Deployment instructions: [DEPLOYMENT.md](DEPLOYMENT.md).
