"""CLI de objetivos residuales: comprobación por defecto y un único proceso por destino."""

import fcntl
import json
import os

import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from scripts import prepare_corpus_targets as cli


@pytest.fixture
def sources(tmp_path):
    manifest = tmp_path / "materialized.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema_version=1,
                kind="materialized_corpus",
                assets=[dict(market="US", symbol="A"), dict(market="CN", symbol="B")],
            )
        )
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    return manifest, prepared, tmp_path / "work" / "supervised"


def arguments(manifest, prepared, output, *extra):
    return ["--manifest", str(manifest), "--prepared", str(prepared), "--output", str(output)] + [
        *extra
    ]


def test_default_mode_only_checks_and_writes_nothing(sources, capsys, monkeypatch):
    monkeypatch.setattr(cli, "prepare_corpus_targets", lambda *a, **k: pytest.fail("Ejecutó"))
    assert cli.main(arguments(*sources)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "check" and report["targets_written"] is False
    assert report["assets"] == 2 and report["markets"] == dict(US=1, CN=1)
    assert report["lock"] == "free" and report["receipts"] == 0
    assert not sources[2].parent.exists()


def test_check_rejects_a_policy_that_the_manifest_does_not_declare(sources):
    with pytest.raises(ValueError):
        cli.check(*sources, input_policy=HISTORICAL_MASKED)


def hold(output):
    path = cli.lock_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return descriptor


def test_a_second_process_is_refused_while_the_destination_is_held(sources, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "prepare_corpus_targets", lambda *a, **k: calls.append(a))
    descriptor = hold(sources[2])
    try:
        assert cli.main(arguments(*sources)) == cli.BUSY
        assert json.loads(capsys.readouterr().out)["lock"] == "busy"
        assert cli.main(arguments(*sources, "--execute")) == cli.BUSY
        assert json.loads(capsys.readouterr().out) == dict(mode="execute", lock="busy")
    finally:
        os.close(descriptor)
    assert calls == []


def test_execute_holds_the_lock_for_the_whole_preparation(sources, capsys, monkeypatch):
    manifest, prepared, output = sources
    seen = []

    def prepare(*args, **options):
        seen.append((args, options, cli._lock_state(cli.lock_path(output))))
        return dict(
            counts=dict(train=3, validation=1),
            assets=[{}, {}],
            reused_assets=1,
            final_test_opened=False,
        )

    monkeypatch.setattr(cli, "prepare_corpus_targets", prepare)
    assert cli.main(arguments(*sources, "--execute", "--backend", "reference")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == dict(
        mode="execute",
        input_policy="strict_inputs_v1",
        counts=dict(train=3, validation=1),
        assets=2,
        reused_assets=1,
        final_test_opened=False,
    )
    assert seen == [
        (
            (manifest, prepared, output),
            dict(backend="reference", input_policy="strict_inputs_v1", target_factors=None),
            "busy",
        )
    ]
    assert cli._lock_state(cli.lock_path(output)) == "free"
