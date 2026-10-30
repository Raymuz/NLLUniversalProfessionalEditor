# NLL V5 - safe GitHub push from Google Colab
# Preserves remote history and does not delete unrelated files.
import os, sys, subprocess, shutil, tempfile, pathlib, getpass, stat
from pathlib import Path

REPO = "https://github.com/Raymuz/NLLUniversalProfessionalEditor.git"
SOURCE = Path(__file__).resolve().parent

def run(args, cwd=None, env=None):
    return subprocess.run(args, cwd=cwd, env=env, check=True, text=True, capture_output=True)

def main():
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        token = None
    if not token:
        token = getpass.getpass("GitHub token with Contents read/write for this repo: ")
    if not token:
        raise RuntimeError("GitHub token required; none provided")
    author = input("Git commit author name [Raymuz]: ").strip() or "Raymuz"
    email = input("Git commit email [noreply@users.noreply.github.com]: ").strip() or "noreply@users.noreply.github.com"
    with tempfile.TemporaryDirectory(prefix="nll-github-") as temp:
        tmp = Path(temp)
        ask = tmp/"askpass.py"
        ask.write_text("#!/usr/bin/env python3\nimport os,sys\nprompt=sys.argv[1].lower() if len(sys.argv)>1 else ''\nprint(os.environ.get('GH_AUTH_TOKEN','') if 'password' in prompt else 'x-access-token')\n")
        ask.chmod(0o700)
        env = os.environ.copy()
        env.update(GIT_ASKPASS=str(ask), GH_AUTH_TOKEN=token, GIT_TERMINAL_PROMPT="0")
        checkout=tmp/"repo"
        run(["git","clone", "--depth", "1", REPO, str(checkout)], env=env)
        # Copy new source without deleting unrelated remote files or history.
        excluded={"github_deploy_colab.py",".git","__pycache__",".ipynb_checkpoints"}
        added=[]
        for f in SOURCE.rglob("*"):
            if not f.is_file() or any(part in excluded for part in f.relative_to(SOURCE).parts):
                continue
            rel=f.relative_to(SOURCE)
            dest=checkout/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(f,dest)
            added.append(str(rel))
        run(["git","config","user.name",author],cwd=checkout)
        run(["git","config","user.email",email],cwd=checkout)
        run(["git","add","--",*added],cwd=checkout)
        status=run(["git","status","--short"],cwd=checkout).stdout
        print("Changed files:\n",status or "(none)")
        if not status.strip():
            print("No changes to publish. Current revision:",run(["git","rev-parse","HEAD"],cwd=checkout).stdout.strip())
            return
        run(["git","commit","-m","Deploy NLL V5 deterministic professional editor source"],cwd=checkout)
        result=run(["git","push","origin","HEAD"],cwd=checkout,env=env)
        sha=run(["git","rev-parse","HEAD"],cwd=checkout).stdout.strip()
        print("[PASS] GitHub push accepted. Commit:",sha)
        print("Repository: https://github.com/Raymuz/NLLUniversalProfessionalEditor")
        print("NOTE: Source published; HF production not changed by this action.")

if __name__ == "__main__":
    main()
