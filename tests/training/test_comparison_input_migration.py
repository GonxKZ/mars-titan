"""Traspaso explícito de una preparación sin actualizaciones a la revisión de entradas."""

import json
import subprocess
from pathlib import Path

import pytest
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training import comparison_campaign as campaign
from mars_titan.training.checkpoints import load_training_state, save_training_state


def checkout_release(root, commit):
    """Ejercitar una revisión publicada aun cuando el checkout de desarrollo haya avanzado."""
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", campaign.ROOT, root], check=True
    )
    subprocess.run(
        ["git", "-C", root, "sparse-checkout", "set", "src/mars_titan", "data/catalogs"],
        check=True,
    )
    subprocess.run(["git", "-C", root, "checkout", "--quiet", "--detach", commit], check=True)
    return root


@pytest.fixture
def packed_runtime(tmp_path):
    from mars_titan.training.comparison_input_migration import PACKED_RELEASE

    return checkout_release(tmp_path / "packed-runtime", PACKED_RELEASE)


def fixture(folder, step=0):
    from mars_titan.training import comparison_input_migration as migration

    module = "mars_titan.training.financial_run"
    legacy = migration.git_digest(campaign.ROOT, migration.WARMUP_RELEASE, module)
    cursor = dict(epoch=0, phase="validation")
    state = dict(
        global_step=step,
        model={"weight": torch.tensor([-0.0, 1.0])},
        optimizer={"state": {}, "param_groups": []},
        rng={"cpu": torch.Generator().manual_seed(17).get_state()},
        selection=None,
        history=[],
        train_metrics=None,
        run=None,
        cursor=cursor,
    )
    ident = dict(implementation={module: legacy}, initial_parameters_sha256="d" * 64, seed=17)
    ck = save_training_state(folder / "fit/checkpoints", state, identity=ident)
    fit = dict(
        kind="titans_financial_chronological_run",
        status="paused",
        final_test_opened=False,
        run_id=migration.run_id(ident),
        identity=ident,
        global_step=step,
        cursor=cursor,
        selection=None,
        history=[],
        recovery_checkpoint=dict(path="checkpoints/" + ck.name, sha256=sha256(ck)),
    )
    atomic_json(folder / "fit/run.json", fit)
    req = dict(code={module: legacy}, seed=17)
    identity = dict(request=req, fit_run_id=fit["run_id"])
    outer = dict(
        kind="titans_walk_forward_window",
        status="paused",
        final_test_opened=False,
        request=req,
        identity=identity,
        run_id=migration.run_id(identity),
        predictions={},
    )
    atomic_json(folder / "run.json", outer)
    return state, fit, outer


def test_zero_step_keeps_the_exact_optimizer_rng_and_weights(tmp_path, packed_runtime):
    from mars_titan.training import comparison_input_migration as migration

    old, new = tmp_path / "original", tmp_path / "target"
    state, fit, report = fixture(old)
    receipt = migration.transfer_initial(old, new, packed_runtime, host_flags=False)
    migrated = json.loads((new / "fit/run.json").read_text())
    outer = json.loads((new / "run.json").read_text())
    restored = load_training_state(
        new / "fit/checkpoints",
        expected_identity=migrated["identity"],
        expected_sha256=migrated["recovery_checkpoint"]["sha256"],
    )
    assert migration.same_state(state, restored)
    assert torch.signbit(restored["model"]["weight"][0])
    assert receipt["source_checkpoint_sha256"] == fit["recovery_checkpoint"]["sha256"]
    assert json.loads((old / "run.json").read_text()) == report
    assert outer["run_id"] == migration.run_id(outer["identity"])
    assert outer["identity"]["fit_run_id"] == migrated["run_id"]
    assert outer["request"]["code"] == outer["identity"]["request"]["code"]
    assert (
        migrated["identity"]["initial_parameters_sha256"]
        == fit["identity"]["initial_parameters_sha256"]
    )
    assert "mars_titan.memory.event_archive" in migrated["identity"]["implementation"]


def test_positive_step_or_unqualified_digest_is_rejected(tmp_path):
    from mars_titan.training import comparison_input_migration as migration

    old, new = tmp_path / "original", tmp_path / "target"
    fixture(old, step=1)
    with pytest.raises(ValueError, match="paso cero"):
        migration.transfer_initial(old, new, Path(campaign.ROOT), host_flags=False)
    assert not new.exists()
    assert not migration.equivalent(
        Path(campaign.ROOT),
        "mars_titan.training.financial_run",
        "0" * 64,
        sha256(Path(campaign.ROOT) / "src/mars_titan/training/financial_run.py"),
    )


def test_catalog_identity_and_explicit_canonical_mode(tmp_path, packed_runtime):
    from mars_titan.training import comparison_input_migration as migration

    root = packed_runtime
    catalog = "data/catalogs/macro-indicators.csv"
    fingerprint = sha256(root / catalog)
    assert migration._translate(root, {catalog: fingerprint}) == {catalog: fingerprint}
    with pytest.raises(ValueError, match="cualificado"):
        migration._translate(root, {catalog: "1" * 64})
    with pytest.raises(ValueError, match="módulo"):
        migration._translate(root, {"../secreto": fingerprint})
    old, new = tmp_path / "source", tmp_path / "target"
    fixture(old)
    result = migration.transfer_initial(old, new, root, archived=False, host_flags=True)
    ident = json.loads((new / "fit/run.json").read_text())["identity"]
    assert not result["archived_inputs"]
    assert "mars_titan.memory.event_archive" not in ident["implementation"]
    assert "mars_titan.models.titans.host_finite" in ident["implementation"]


def test_modified_destination_code_rejects_a_real_step_zero(tmp_path, packed_runtime):
    from mars_titan.training import comparison_input_migration as migration

    original, target = tmp_path / "original", tmp_path / "target"
    fixture(original)
    numerical = packed_runtime / "src/mars_titan/training/financial_run.py"
    numerical.write_bytes(numerical.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="no está cualificado"):
        migration.transfer_initial(original, target, packed_runtime, host_flags=False)
    assert not target.exists()
