"""Deploy this repo as a Docker Space on Hugging Face (one step).

    python scripts/deploy_hf.py --token-file PATH --space OWNER/incident-intel

The token is read from the file (first token that starts with hf_), used in memory only and
never printed or written anywhere. Only the files the container needs are uploaded.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FRONT_MATTER = """---
title: Incident Intelligence
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Find the unusual moments in fixed-camera video
---

Source, benchmark and limits: https://github.com/Professional50coder/incident-intel
"""


def read_token(path: Path) -> str:
    m = re.search(r"hf_[A-Za-z0-9]{20,}", path.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        sys.exit("no hf_ token found in the token file")
    return m.group(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file", type=Path, required=True)
    ap.add_argument("--space", required=True, help="OWNER/NAME of the Space")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi(token=read_token(args.token_file))
    who = api.whoami()["name"]
    print("authenticated as", who)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=str(ROOT),
        allow_patterns=["Dockerfile", "requirements.txt", "pyproject.toml", "LICENSE", "src/**", "checkpoints/*.pt", "samples/*"],
        ignore_patterns=["**/__pycache__/**", "*.pyc", "*.egg-info/**"],
        commit_message="Deploy",
    )
    with tempfile.TemporaryDirectory() as td:
        readme = Path(td) / "README.md"
        readme.write_text(FRONT_MATTER, encoding="utf-8")
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=args.space,
                        repo_type="space", commit_message="Space card")
    owner, name = args.space.split("/")
    print(f"deployed: https://huggingface.co/spaces/{args.space}")
    print(f"app url:  https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
