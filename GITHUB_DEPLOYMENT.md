# GitHub deployment

Repository: https://github.com/Raymuz/NLLUniversalProfessionalEditor

This is the source repository for the NLL V5 Docker/Gradio/FastAPI/Flask release. It is not itself an app hosting environment. The Hugging Face Space can be deployed separately from the same source.

## Source preservation

Never use `git push --force` or delete existing remote branches. Back up or review the existing repository before merging deployment files. A release is not production verified until live API and UI tests pass.

## Deployment via Colab

Unzip this archive in `/content/NLL_GitHub_Package` and run `github_deploy_colab.py`. Set `GITHUB_TOKEN` in Colab Secrets (a fine-grained token with Contents read/write access to this repository) or provide it securely when prompted. The script clones the existing remote's default branch, copies the application files without removing unrelated paths, commits only changed files, and pushes. It aborts on non-fast-forward rather than overwriting remote work.

## Verification

1. Confirm repository files are visible on GitHub.
2. Check GitHub commit hash.
3. Independently build and test the Docker app or deploy to your Hugging Face Space.
