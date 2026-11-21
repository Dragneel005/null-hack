#!/usr/bin/env python3
"""Deploy the Push-up Form Coach to a Hugging Face Docker Space (free CPU tier).

Usage (from the repo root):
    HF_TOKEN=hf_xxx .venv/bin/python deploy/deploy_hf.py <space_name>            # upload + wait for build
    HF_TOKEN=hf_xxx .venv/bin/python deploy/deploy_hf.py <space_name> --no-wait  # upload only
    HF_TOKEN=hf_xxx .venv/bin/python deploy/deploy_hf.py <space_name> --status   # just poll build status
    .venv/bin/python deploy/deploy_hf.py <space_name> --dry-run                  # build staging dir only

Optional: set ANTHROPIC_API_KEY in the environment and it is stored as a Space secret
(the app falls back to its offline coach without it).

HF builds the Dockerfile remotely, so Docker is not needed locally. Only the files the app
needs at runtime are uploaded (never .env, secrets.toml, outputs/, tests/ or the GitHub README).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "deploy" / "_staging"
TEMPLATE = ROOT / "deploy" / "space_README.md"
FORBIDDEN = (".env", "secrets.toml", "outputs", ".venv", ".git", "tests")
MAX_STAGING_MB = 200


def _hf():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    import huggingface_hub
    return huggingface_hub


def app_url(user: str, space: str) -> str:
    sub = re.sub(r"[^a-z0-9-]", "-", f"{user}-{space}".lower())
    return f"https://{sub}.hf.space"


def build_staging() -> Path:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    files = sorted(ROOT.glob("*.py"))
    files += sorted((ROOT / "models").glob("*.task"))
    files += sorted(p for p in (ROOT / "samples").rglob("*") if p.is_file())
    files += [ROOT / ".streamlit" / "config.toml", ROOT / "Dockerfile",
              ROOT / "requirements.txt", ROOT / ".dockerignore"]
    for src in files:
        if not src.exists():
            continue
        rel = src.relative_to(ROOT)
        if "__pycache__" in rel.parts:
            continue
        dst = STAGING / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Space README = HF frontmatter template (NOT the GitHub README).
    shutil.copy2(TEMPLATE, STAGING / "README.md")

    # Safety checks: no secrets / scratch output, sane size, model present.
    bad = [p for p in STAGING.rglob("*") if any(part in FORBIDDEN for part in p.relative_to(STAGING).parts)]
    if bad:
        sys.exit(f"Refusing to deploy, forbidden files staged: {bad}")
    if not list((STAGING / "models").glob("*.task")):
        sys.exit("No MediaPipe model in models/*.task - run models/download.sh first.")
    size_mb = sum(p.stat().st_size for p in STAGING.rglob("*") if p.is_file()) / 1e6
    n = sum(1 for p in STAGING.rglob("*") if p.is_file())
    print(f"Staged {n} files, {size_mb:.1f} MB -> {STAGING}")
    if size_mb > MAX_STAGING_MB:
        sys.exit(f"Staging dir too big ({size_mb:.0f} MB > {MAX_STAGING_MB} MB)")
    return STAGING


def tail_build_logs(repo_id: str, token: str, seconds: float = 15, lines: int = 40) -> None:
    """Best effort: print the last lines of the Space build log (server-sent events stream)."""
    url = f"https://huggingface.co/api/spaces/{repo_id}/logs/build"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    out: list[str] = []
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=seconds) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    m = re.search(r'"data"\s*:\s*"(.*)"', line)
                    out.append((m.group(1) if m else line[5:]).replace("\\n", "").replace('\\"', '"'))
                if time.time() - t0 > seconds:
                    break
    except Exception as e:  # timeouts are expected on a live stream
        if not out:
            print(f"(could not fetch build logs: {e})")
    if out:
        print("---- last build log lines ----")
        print("\n".join(out[-lines:]))


def wait_for_space(api, repo_id: str, token: str, timeout_min: float = 25, expect_rebuild: bool = False) -> bool:
    print(f"Polling build status of {repo_id} (Ctrl+C to stop) ...")
    t0, last, seen_build = time.time(), None, not expect_rebuild
    while time.time() - t0 < timeout_min * 60:
        try:
            rt = api.get_space_runtime(repo_id)
            stage = str(getattr(rt.stage, "value", rt.stage))
        except Exception as e:
            stage = f"UNKNOWN ({e.__class__.__name__})"
        if stage != last:
            print(f"[{time.time() - t0:5.0f}s] stage: {stage}")
            last = stage
        if "BUILDING" in stage or "STARTING" in stage:
            seen_build = True
        if stage == "RUNNING" and seen_build:
            print("Space is RUNNING.")
            return True
        if stage == "RUNNING" and time.time() - t0 > 90:  # rebuild never started; it's up anyway
            print("Space is RUNNING.")
            return True
        # Right after an upload HF can still report the OLD stage (e.g. NO_APP_FILE on a fresh
        # Space, or a previous BUILD_ERROR) for a few seconds: give it 60 s before failing.
        stale_ok = not seen_build and time.time() - t0 < 60
        if not stale_ok and ("ERROR" in stage or stage in ("NO_APP_FILE", "PAUSED", "STOPPED")):
            print(f"Space failed: {stage}. Logs: https://huggingface.co/spaces/{repo_id}?logs=build")
            tail_build_logs(repo_id, token)
            return False
        time.sleep(10)
    print("Timed out waiting; check the Space page.")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("space_name", help="Space name, e.g. pushup-form-coach")
    ap.add_argument("--status", action="store_true", help="only poll build status, no upload")
    ap.add_argument("--no-wait", action="store_true", help="upload and exit without polling")
    ap.add_argument("--dry-run", action="store_true", help="build deploy/_staging only, no network")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    a = ap.parse_args()

    if a.dry_run:
        build_staging()
        return 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN (a *write* token from https://huggingface.co/settings/tokens).")

    hh = _hf()
    api = hh.HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{a.space_name}"
    space_url = f"https://huggingface.co/spaces/{repo_id}"

    if a.status:
        ok = wait_for_space(api, repo_id, token)
        print(f"Space page: {space_url}\nApp URL:    {app_url(user, a.space_name)}")
        return 0 if ok else 1

    staging = build_staging()
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    exist_ok=True, private=a.private)

    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        api.add_space_secret(repo_id, "ANTHROPIC_API_KEY", key)
        print("Set Space secret ANTHROPIC_API_KEY.")
    else:
        print("ANTHROPIC_API_KEY not set - the app will use its offline coach.")

    print(f"Uploading to {repo_id} ...")
    api.upload_folder(
        repo_id=repo_id, repo_type="space", folder_path=str(staging),
        commit_message="Deploy push-up form coach",
        delete_patterns=["*.py", "samples/*", "models/*", "*.txt", ".streamlit/*"],  # drop stale files
    )
    print(f"Uploaded.\nSpace page: {space_url}\nApp URL:    {app_url(user, a.space_name)}")

    if a.no_wait:
        print(f"Check build later with: .venv/bin/python deploy/deploy_hf.py {a.space_name} --status")
        return 0
    ok = wait_for_space(api, repo_id, token, expect_rebuild=True)
    print(f"Space page: {space_url}\nApp URL:    {app_url(user, a.space_name)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
