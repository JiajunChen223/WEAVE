"""Verify upstream commit and file hashes."""
import hashlib
import json
import subprocess
from pathlib import Path


def verify_source(method, folder):
    records = json.loads(Path(__file__).with_name("sources.json").read_text(encoding="utf-8"))
    if method not in records or not records[method].get("commit"):
        raise ValueError("This method has no pinned upstream checkout in the registry")
    record, folder = records[method], Path(folder).resolve()
    result = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True)
    if result.stdout.strip() != record["commit"]:
        raise ValueError("Upstream commit mismatch")
    for relative, expected in record.get("content_sha256", {}).items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError("Missing or invalid pinned source file: " + relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Upstream content differs: " + relative)
    return record
