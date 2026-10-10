#!/usr/bin/env python3
"""Prepare both paper corpora from checksum-pinned Project Gutenberg eBook #10.

No corpus bytes are distributed in this repository. Downloaded source bytes,
prepared token streams, token counts and prefix counts must match the committed
manifests. A changed upstream edition is rejected before any output is written.
Existing verified outputs are preserved; existing different bytes are rejected.

Usage: python scripts/get_kjv.py [--from-file pg10.txt] [--out-directory data]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request


START = "*** START OF THE PROJECT GUTENBERG EBOOK"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"
HEADING = "The Old Testament of the King James Version of the Bible"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def body(raw):
    start = raw.index(START)
    start = raw.index("\n", start) + 1
    end = raw.index(END, start)
    return raw[start:end]


def preprocess(raw):
    """Canonical, case-preserving word/punctuation tokens used by the paper."""
    text = body(raw)
    text = re.sub(r"\d+:\d+", " ", text)
    text = re.sub(r"’(?!s)", " ", text)
    text = text.replace("*", " ").replace("-", " ")
    return re.findall(r"[A-Za-z]+|’s|[^\sA-Za-z]", text)


def secondary(raw):
    """Explicit secondary formatting and Unicode word/punctuation control."""
    text = body(raw).lstrip()
    if not text.startswith(HEADING):
        raise ValueError("secondary source must have the exact opening heading")
    text = text[len(HEADING):]
    if text.count("--") != 1:
        raise ValueError("secondary source must contain exactly one double hyphen")
    text = text.replace("--", "—")
    separators = list(re.finditer(r"\*\s*\*\s*\*", text))
    if len(separators) != 1 or text.count("*") != 3:
        raise ValueError("secondary source must contain one three-star separator")
    text = re.sub(r"\*\s*\*\s*\*", " ", text)
    text = re.sub(r"\d+:\d+", " ", text)
    return re.findall(r"\w+|[^\w\s]", text) + ["*", "*", "*"]


def validate(data, manifest):
    if digest(data) != manifest["sha256"]:
        raise ValueError(f"prepared checksum differs for {manifest['dataset_id']}")
    tokens = data.decode("utf8").split()
    if len(tokens) != manifest["tokens"] or len(set(tokens)) != manifest["distinct_types"]:
        raise ValueError("prepared token/type counts differ")
    if "bytes" in manifest and len(data) != manifest["bytes"]:
        raise ValueError("prepared byte count differs")
    for count, expected in manifest.get("prefix_distinct_types", {}).items():
        if len(set(tokens[:int(count)])) != expected:
            raise ValueError("prepared prefix type count differs")


def prepare(raw, manifests, destination):
    """Verify both streams before writing either; never replace existing files."""
    for manifest in manifests:
        if digest(raw) not in manifest["source_sha256"]:
            raise ValueError("downloaded source checksum differs; no outputs written")
    text = raw.decode("utf8")
    payloads = [(" ".join(preprocess(text))).encode("utf8"),
                (" ".join(secondary(text)) + "\n").encode("utf8")]
    paths = []
    for data, manifest in zip(payloads, manifests, strict=True):
        validate(data, manifest)
        path = Path(destination) / Path(manifest["file"]).name
        if path.is_symlink() or (path.exists() and path.read_bytes() != data):
            raise FileExistsError(f"refusing to replace existing corpus: {path}")
        paths.append(path)
    Path(destination).mkdir(parents=True, exist_ok=True)
    for path, data in zip(paths, payloads, strict=True):
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
    return [{"path": str(path), "sha256": digest(data), "bytes": len(data)}
            for path, data in zip(paths, payloads, strict=True)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-file", type=Path, help="checksum-pinned original Gutenberg text")
    parser.add_argument("--out-directory", type=Path,
                        help="prepared files directory (default: repository data/)")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    manifests = [json.loads((root / "data" / f"{name}.manifest.json").read_text())
                 for name in ("kjv", "kjv-secondary")]
    if args.from_file:
        raw = args.from_file.read_bytes()
        source = str(args.from_file)
    else:
        errors = []
        for source in manifests[0]["source_urls"]:
            try:
                with urllib.request.urlopen(source, timeout=60) as response:
                    raw = response.read()
                if digest(raw) not in manifests[0]["source_sha256"]:
                    raise ValueError("source checksum changed")
                break
            except (OSError, ValueError) as error:
                errors.append(f"{source}: {error}")
        else:
            raise RuntimeError("No matching source; use --from-file with the pinned edition. "
                               + "; ".join(errors))
    outputs = prepare(raw, manifests, args.out_directory or root / "data")
    print(json.dumps({"source": source, "source_sha256": digest(raw), "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
