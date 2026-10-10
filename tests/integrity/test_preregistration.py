"""Cadena de registro previo: alteraciones, desviaciones y prueba externa con git."""

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from mars_titan.data.storage import sha256
from mars_titan.integrity import preregistration as registry_module

MOMENT = datetime(2026, 10, 10, 9, 0, tzinfo=UTC)


def document(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(status="declared_before_evaluation", **values)))
    return path


def report(path, configuration, created):
    path.write_text(
        json.dumps(dict(configuration=dict(sha256=configuration), created_at_utc=created))
    )
    return path


@pytest.fixture
def chain(tmp_path):
    registry = tmp_path / "registry.jsonl"
    config = document(tmp_path / "config.json", metric="mae")
    first = registry_module.declare(registry, config, note="Comparación principal", now=MOMENT)
    revised = document(tmp_path / "config-v2.json", metric="mae", replicates=2000)
    later = MOMENT + timedelta(hours=1)
    change = registry_module.deviate(
        registry, first["entry_sha256"], reason="Más réplicas", replacement=revised, now=later
    )
    return registry, config, first, change


def test_a_declaration_and_its_deviation_form_a_valid_chain(chain):
    registry, config, first, change = chain
    entries = registry_module.read_chain(registry)
    assert [e["sequence"] for e in entries] == [0, 1]
    assert entries[0]["previous_sha256"] == registry_module.GENESIS
    assert entries[1]["previous_sha256"] == first["entry_sha256"]
    assert entries[0]["subject_sha256"] == sha256(config)
    assert change["declaration_sha256"] == first["entry_sha256"]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: [lines[0].replace("principal", "secundaria"), lines[1]], "alterada"),
        (lambda lines: lines[1:], "orden"),
        (lambda lines: [lines[1], lines[0]], "orden"),
    ],
)
def test_editing_removing_or_reordering_an_entry_breaks_the_chain(chain, edit, message):
    registry = chain[0]
    lines = registry.read_text().splitlines()
    registry.write_text("\n".join(edit(lines)) + "\n")
    with pytest.raises(ValueError, match=message):
        registry_module.read_chain(registry)


def test_a_rehashed_edit_still_breaks_the_link_to_the_next_entry(chain):
    registry = chain[0]
    entries = [json.loads(line) for line in registry.read_text().splitlines()]
    entries[0]["note"] = "Otra nota"
    entries[0]["entry_sha256"] = registry_module._entry_sha256(entries[0])
    registry.write_text("".join(registry_module._canonical(e) + "\n" for e in entries))
    with pytest.raises(ValueError, match="rompe la cadena"):
        registry_module.read_chain(registry)


def test_invalid_entries_are_refused_before_writing(chain, tmp_path):
    registry, config, *_ = chain
    before = registry.read_bytes()
    with pytest.raises(ValueError, match="no apunta"):
        registry_module.deviate(
            registry, "f" * 64, reason="Sin declaración", now=MOMENT + timedelta(days=1)
        )
    with pytest.raises(ValueError, match="dos veces"):
        registry_module.declare(registry, config, note="Repetida", now=MOMENT + timedelta(days=1))
    with pytest.raises(ValueError, match="fecha"):
        other = document(tmp_path / "other.json", metric="mse")
        registry_module.declare(registry, other, note="Fecha previa", now=MOMENT)
    assert registry.read_bytes() == before


def test_a_report_after_its_declaration_passes_and_lists_deviations(chain, tmp_path):
    registry, config, first, _ = chain
    later = (MOMENT + timedelta(days=1)).isoformat()
    result = registry_module.check_report(
        registry, report(tmp_path / "report.json", sha256(config), later)
    )
    assert result["passed"] and result["declaration_sha256"] == first["entry_sha256"]
    assert [d["reason"] for d in result["deviations"]] == ["Más réplicas"]


def test_a_report_before_its_declaration_or_undeclared_fails(chain, tmp_path):
    registry, config, *_ = chain
    early = (MOMENT - timedelta(minutes=1)).isoformat()
    result = registry_module.check_report(
        registry, report(tmp_path / "early.json", sha256(config), early)
    )
    assert not result["passed"] and result["declared_before_report"] is False
    unknown = registry_module.check_report(
        registry, report(tmp_path / "unknown.json", "a" * 64, MOMENT.isoformat())
    )
    assert not unknown["passed"] and unknown["declared"] is False


def git(repository, *arguments, when=None):
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="Registro",
        GIT_AUTHOR_EMAIL="registro@example.invalid",
        GIT_COMMITTER_NAME="Registro",
        GIT_COMMITTER_EMAIL="registro@example.invalid",
    )
    if when is not None:
        env.update(GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )


def test_the_commit_that_adds_the_declaration_is_the_external_evidence(tmp_path):
    repository, remote = tmp_path / "repo", tmp_path / "remote.git"
    repository.mkdir()
    git(repository, "init", "-q", "-b", "main")
    git(tmp_path, "init", "-q", "--bare", str(remote))
    config = document(repository / "configs" / "comparison.json", metric="mae")
    registry = repository / "configs" / "preregistration.jsonl"
    entry = registry_module.declare(registry, config, note="Principal", root=repository, now=MOMENT)
    assert entry["subject_path"] == "configs/comparison.json"
    git(repository, "add", ".")
    git(repository, "commit", "-q", "-m", "declare", when="2026-10-10T10:00:00+00:00")
    after = report(tmp_path / "after.json", sha256(config), "2026-10-11T00:00:00+00:00")
    local = registry_module.check_report(registry, after, repository=repository)
    assert local["passed"] and local["committed_before_report"]
    assert local["published_before_report"] is False
    git(repository, "remote", "add", "origin", str(remote))
    git(repository, "push", "-q", "origin", "main")
    git(repository, "fetch", "-q", "origin")
    published = registry_module.check_report(registry, after, repository=repository)
    assert published["published_before_report"] is True
    # Declarada a las 09:00 en local, pero con el commit posterior al informe de las 09:30.
    before = report(tmp_path / "before.json", sha256(config), "2026-10-10T09:30:00+00:00")
    late = registry_module.check_report(registry, before, repository=repository)
    assert late["declared_before_report"] and not late["committed_before_report"]
    assert not late["passed"]


def test_the_cli_declares_and_checks(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = document(tmp_path / "config.json", metric="mae")
    registry = tmp_path / "registry.jsonl"
    assert registry_module.main(["declare", str(registry), str(config), "--note", "CLI"]) == 0
    declared = capsys.readouterr().out.strip()
    assert registry_module.read_chain(registry)[0]["entry_sha256"] == declared
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    target = tmp_path / "check.json"
    arguments = ["check", str(registry), str(report(tmp_path / "r.json", sha256(config), future))]
    assert registry_module.main([*arguments, "--output", str(target)]) == 0
    assert json.loads(target.read_text())["passed"] is True
