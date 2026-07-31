from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "assets" / "models" / "htdemucs_6s"
SOURCE_MANIFEST = DEFAULT_MODEL_DIR / "model-source.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fetch_model(output_dir: Path) -> Path:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source["checkpoint"]
    expected_prefix = str(source["sha256Prefix"]).lower()

    if destination.is_file():
        digest = sha256_file(destination)
        if digest.startswith(expected_prefix):
            _write_ready_manifest(output_dir, source, digest)
            return destination
        destination.unlink()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".part",
        dir=output_dir,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(
            source["url"],
            headers={"User-Agent": "SkyAutoMusic-build/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        digest = sha256_file(temporary)
        if not digest.startswith(expected_prefix):
            raise RuntimeError(
                f"Model checksum mismatch: expected {expected_prefix}, got {digest}"
            )
        os.replace(temporary, destination)
        _write_ready_manifest(output_dir, source, digest)
        return destination
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _write_ready_manifest(output_dir: Path, source: dict, digest: str) -> None:
    ready = {
        "name": source["name"],
        "checkpoint": source["checkpoint"],
        "sha256": digest,
        "source": source["url"],
        "license": source["license"],
    }
    path = output_dir / "model-manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(ready, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    args = parser.parse_args()
    path = fetch_model(args.output_dir.resolve())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

