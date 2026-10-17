"""Helpers shared by client.run (M1) and client.load (M2)."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import httpx


def code_sha256() -> str:
    """Hash of the client code that actually ran: identifies the version even for dirty runs."""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def git_state() -> dict:
    # In-cluster runs have no .git: the launcher passes the state in via env.
    if commit := os.environ.get("CLIENT_GIT_COMMIT"):
        return {
            "commit": commit,
            "dirty": os.environ.get("CLIENT_GIT_DIRTY") == "1",
            "client_code_sha256": code_sha256(),
        }

    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
        "client_code_sha256": code_sha256(),
    }


async def server_info(client: httpx.AsyncClient) -> dict:
    info: dict = {}
    for key, path in (("version", "/version"), ("models", "/v1/models")):
        try:
            r = await client.get(path)
            info[key] = r.json() if r.status_code == 200 else {"http_status": r.status_code}
        except (httpx.HTTPError, ValueError) as e:
            info[key] = {"error": str(e)}
    return info
