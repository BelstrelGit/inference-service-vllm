"""Helpers shared by client.run (M1) and client.load (M2)."""

from __future__ import annotations

import os
import subprocess

import httpx


def git_state() -> dict:
    # In-cluster runs have no .git: the launcher passes the state in via env.
    if commit := os.environ.get("CLIENT_GIT_COMMIT"):
        return {"commit": commit, "dirty": os.environ.get("CLIENT_GIT_DIRTY") == "1"}

    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
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
