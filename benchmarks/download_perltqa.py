"""Download the pinned official PerLTQA English v2 holdout files."""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path


UPSTREAM_COMMIT = "8d9e19868e239740ef701e603ec205cd581f221b"
FILES = {
    "perltmem_en_v2.json": {
        "url": (
            "https://raw.githubusercontent.com/Elvin-Yiming-Du/PerLTQA/"
            f"{UPSTREAM_COMMIT}/Dataset/en_v2/perltmem_en_v2.json"
        ),
        "sha256": "fb3011d78babdc9c5323a8d303ca7d4cdb9e2e08992c9890f9ccb2362ff8be94",
    },
    "perltqa_en_v2.json": {
        "url": (
            "https://raw.githubusercontent.com/Elvin-Yiming-Du/PerLTQA/"
            f"{UPSTREAM_COMMIT}/Dataset/en_v2/perltqa_en_v2.json"
        ),
        "sha256": "ca9f29cbb23eb8f7dbfb792359d9ff90eec066e6d927c83f6c0b7d0bf7baff23",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(data_dir: Path | None = None) -> tuple[Path, Path]:
    """Return verified memory and QA paths, downloading missing files."""
    target_dir = data_dir or Path(__file__).parent / "data"
    target_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, Path] = {}
    for filename, metadata in FILES.items():
        destination = target_dir / filename
        if destination.exists():
            actual = _sha256(destination)
            if actual != metadata["sha256"]:
                raise ValueError(
                    f"Refusing to replace {destination}: expected SHA-256 "
                    f"{metadata['sha256']}, found {actual}"
                )
        else:
            with tempfile.NamedTemporaryFile(
                dir=target_dir, prefix=f".{filename}.", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                urllib.request.urlretrieve(metadata["url"], temporary_path)
                actual = _sha256(temporary_path)
                if actual != metadata["sha256"]:
                    raise ValueError(
                        f"Checksum mismatch for {filename}: expected "
                        f"{metadata['sha256']}, found {actual}"
                    )
                temporary_path.replace(destination)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
        resolved[filename] = destination

    return (
        resolved["perltmem_en_v2.json"],
        resolved["perltqa_en_v2.json"],
    )


if __name__ == "__main__":
    memory_path, qa_path = download()
    print(memory_path)
    print(qa_path)
