"""Download LongMemEval benchmark data from HuggingFace."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
BASE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

FILES = {
    "longmemeval_s_cleaned.json": f"{BASE_URL}/longmemeval_s_cleaned.json",
    "longmemeval_oracle.json": f"{BASE_URL}/longmemeval_oracle.json",
}


def download(name: str = "longmemeval_s_cleaned.json"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    if path.exists():
        print(f"Already exists: {path}")
        data = json.loads(path.read_text())
        print(f"  {len(data)} questions loaded")
        return path

    url = FILES[name]
    print(f"Downloading {name} from HuggingFace...")
    urllib.request.urlretrieve(url, path)
    data = json.loads(path.read_text())
    print(f"  Downloaded: {len(data)} questions, {path.stat().st_size / 1e6:.1f} MB")
    return path


if __name__ == "__main__":
    download("longmemeval_s_cleaned.json")
