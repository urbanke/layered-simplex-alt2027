"""Bind a freshly built store to the current paper's exact geometry contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .artifacts import read_json, sha256


def direct_manifest(settings):
    return {"version": "v2", "H": settings["grid_step"], "U_MAX": settings["u_max"]}


def direct_reference_files(settings):
    payload = json.dumps(direct_manifest(settings), indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    return {"manifest.json": hashlib.sha256(payload.encode("utf8")).hexdigest()}


def materialize_store_spec(spec, store, *, repo):
    """Verify all stored bytes, then pin this build without claiming accuracy.

    Existing explicit hash specifications remain supported for diagnostic stores.
    The public paper specification instead fixes geometry and numerical settings;
    new builds receive their own hashes and require fresh numerical validation.
    """
    from .sealed_store_build import create_plan, verify_store

    if "files_sha256" in spec:
        return spec
    if spec.get("format") != "sealed" or spec.get("schema_version") != 1:
        raise ValueError("a sealed geometry contract or explicit file hashes are required")
    geometry = spec.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("sealed specification is missing its geometry contract")
    relative = Path(geometry["anchors_file"])
    root = Path(repo).resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("anchor path must be relative to the release")
    anchor_file = root / relative
    if anchor_file.is_symlink() or not anchor_file.resolve().is_relative_to(root):
        raise ValueError("unsafe anchor path")
    source = read_json(anchor_file)
    anchors = {str(level): source["levels"][str(level)]["anchors"]
               for level in geometry["levels"]}
    expected = create_plan(
        anchors=anchors, levels=geometry["levels"],
        support_max_count=geometry["support_max_count"],
        grid_step=geometry["grid_step"], u_max=geometry["u_max"],
        left_drop=geometry["left_drop"],
    )
    path = Path(store)
    actual = read_json(path / "plan.json")
    keys = ("levels", "anchors", "support_max_count", "grid_step", "u_max",
            "left_drop", "interpolation_degree", "count_degree", "dtype",
            "builder_settings", "source_sha256")
    if any(actual.get(key) != expected[key] for key in keys):
        raise ValueError("built store differs from the declared geometry or numerical builder")
    verified = verify_store(path)
    if sha256(path / "plan.json") != verified["plan_sha256"]:
        raise ValueError("store plan changed during verification")
    hashes = {**verified["files"], "manifest.json": sha256(path / "manifest.json")}
    return {**spec, "files_sha256": hashes}


def require_declared_engine(engine, spec):
    """Preserve all declared defaults; backend selection remains explicit."""
    from .depth import StoreConfig

    actual = asdict(StoreConfig(**engine["store"]))
    expected = asdict(StoreConfig(path=actual["path"],
                                 files_sha256=spec["files_sha256"], **spec["settings"]))
    for key in ("interpolation_backend", "native_library_path", "native_library_sha256"):
        expected[key] = actual[key]
    if (engine.get("mode") != "store" or engine.get("prediction_tolerance", 1e-3) != 1e-3
            or actual != expected):
        raise ValueError("engine differs from declared paper settings or verified store hashes")
