"""Clean archive/clone execution and immutable input preparation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from lsa.alt.artifacts import Run, sha256, source_files, source_identity, verify_run
from lsa.alt.cli import main
from lsa.alt.depth import DepthEvaluator, StoreConfig
from lsa.alt.sealed_store_build import create_plan
from lsa.alt import sealed_store_build as builder
from lsa.alt.store_preparation import materialize_store_spec, require_declared_engine


def release(root):
    (root / "src").mkdir(parents=True)
    (root / "src/model.py").write_text("value = 1\n")
    manifest = {"schema_version": 1, "format": "lsa-release-source-v1",
                "files_sha256": {name: sha256(root / name) for name in source_files(root)}}
    (root / "release-source.json").write_text(json.dumps(manifest))
    return root


def test_archive_relocation_keeps_identity_and_records_complete_run(tmp_path):
    first = release(tmp_path / "one")
    second = tmp_path / "two"
    shutil.copytree(first, second)
    assert source_identity(first) == source_identity(second)
    assert source_identity(first)["source_kind"] == "release-archive"
    assert source_identity(first)["dirty"] is False
    with Run(tmp_path / "run", repo=second, protocol={"status": "frozen"},
             experiment="example", purpose="production") as run:
        (run.path / "result.json").write_text("{}")
    assert verify_run(tmp_path / "run")["status"] == "complete"


def test_prepared_corpus_inputs_do_not_change_release_source_identity(tmp_path):
    root = release(tmp_path / "release")
    before = source_identity(root)
    (root / "data").mkdir()
    for name in ("kjv.txt", "kjv-secondary.txt", "kjv.txt.gz", "kjv-secondary.txt.gz"):
        (root / "data" / name).write_bytes(b"separately checked input")
    assert source_identity(root) == before


@pytest.mark.parametrize("mutation", ["changed", "missing", "added", "symlink"])
def test_archive_integrity_rejects_changed_inventory(tmp_path, mutation):
    root = release(tmp_path / "release")
    path = root / "src/model.py"
    if mutation == "changed":
        path.write_text("value = 2")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "added":
        (root / "src/extra.py").write_text("pass")
    else:
        other = tmp_path / "external.py"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    with pytest.raises(ValueError, match="inventory|checksum|symlink"):
        source_identity(root)


def test_changed_archive_source_leaves_failed_run_record(tmp_path):
    root = release(tmp_path / "release")
    with pytest.raises(RuntimeError, match="source integrity"):
        with Run(tmp_path / "run", repo=root, protocol={},
                 experiment="example", purpose="validation"):
            (root / "src/model.py").write_text("value = 2")
    assert verify_run(tmp_path / "run", require_complete=False)["status"] == "failed"


def test_unsealed_directory_cannot_claim_production_or_parent_git(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    root = tmp_path / "unsealed"
    root.mkdir()
    (root / "source.py").write_text("value = 1")
    assert source_identity(root)["source_kind"] == "unsealed-directory"
    assert source_identity(root)["dirty"] is True
    with pytest.raises(ValueError, match="clean"):
        Run(tmp_path / "run", repo=root, protocol={"status": "frozen"},
            experiment="example", purpose="production")


def test_import_does_not_compile_or_load_reference_library():
    result = subprocess.run([sys.executable, "-c",
        "import lsa, lsa.alt.depth, sys; assert 'lsa.kernel' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_anchor_grid_reproduces_full_store_geometry():
    root = Path(__file__).resolve().parents[1]
    plan = create_plan(root / "experiments/alt2027/count-anchors.json")
    assert plan["estimated_columns"] == 50553
    assert plan["estimated_payload_bytes"] == 11633407536
    assert all(len(rows) == 369 and rows[:257] == list(range(257))
               for rows in plan["anchors"].values())


def test_reference_preparation_preserves_outer_settings_and_uses_direct_low_depths(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "experiments/alt2027/sealed-store-candidate.json").read_text())
    engine = {"mode": "store", "prediction_tolerance": 1e-3,
              "store": {"path": "/not-opened-candidate", "files_sha256": {"manifest.json": "0" * 64},
                        **spec["settings"]}}
    config = tmp_path / "candidate.json"
    config.write_text(json.dumps(engine))
    out = tmp_path / "reference"
    assert main(["--repo", str(root), "configure-reference", "--engine-config",
                 str(config), "--out", str(out)]) == 0
    reference = json.loads((out / "engine.json").read_text())
    assert reference["store"]["saddle_min_depth"] == 2
    for key in ("grid_step", "u_max", "minimum_grid_step", "significance_gap"):
        assert reference["store"][key] == engine["store"][key]
    with DepthEvaluator(mode="store", store=StoreConfig(**reference["store"]),
                        prediction_tolerance=1e-3) as evaluator:
        result = evaluator.evidence_at_depths(10, [2, 1], [0, 1, 2, 9])
        assert result.depths == (0, 1, 2, 9)
        assert len(result.log_evidence) == 4
    with pytest.raises(FileExistsError):
        main(["--repo", str(root), "configure-reference", "--engine-config",
              str(config), "--out", str(out)])


def test_fresh_store_contract_verifies_actual_payload_and_exact_geometry(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    anchors = {"2": list(range(9))}
    (root / "anchors.json").write_text(json.dumps(
        {"levels": {key: {"anchors": value} for key, value in anchors.items()}}))
    plan = builder.create_plan(anchors=anchors, levels=[2], support_max_count=8)
    store = tmp_path / "store"
    builder.write_plan(plan, store)
    builder.build_store(store)
    builder.seal_store(store)
    settings = {"format": "sealed", "max_depth": 2, "max_d": 10,
                "max_n": 5, "max_count": 5}
    spec = {"schema_version": 1, "format": "sealed", "settings": settings,
            "geometry": {"anchors_file": "anchors.json", "levels": [2],
                         "support_max_count": 8, "grid_step": .02,
                         "u_max": 80., "left_drop": 60.}}
    bound = materialize_store_spec(spec, store, repo=root)
    assert set(bound["files_sha256"]) == {
        "plan.json", "manifest.json", "level_002.bin", "level_002.index.json",
        "level_002.complete.json"}
    engine = {"mode": "store", "prediction_tolerance": 1e-3,
              "store": {"path": str(store), **settings,
                        "files_sha256": bound["files_sha256"]}}
    require_declared_engine(engine, bound)
    with DepthEvaluator(mode="store", store=StoreConfig(**engine["store"])) as evaluator:
        result = evaluator.evidence_at_depths(10, [2, 1], [0, 1, 2])
        assert result.depths == (0, 1, 2)
    with pytest.raises(ValueError, match="declared paper settings"):
        require_declared_engine({**engine, "prediction_tolerance": 1e-2}, bound)
    altered = {**spec, "geometry": {**spec["geometry"], "grid_step": .04}}
    with pytest.raises(ValueError, match="geometry"):
        materialize_store_spec(altered, store, repo=root)
    payload = store / "level_002.bin"
    with payload.open("r+b") as stream:
        stream.write(b"damaged!")
    with pytest.raises(ValueError, match="hash"):
        materialize_store_spec(spec, store, repo=root)
