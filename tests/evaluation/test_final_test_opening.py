"""Apertura única del test final con un repositorio temporal y un abridor simulado.

El abridor de estas pruebas solo escribe un archivo de texto: no prepara objetivos de
2024 ni lee datos. La declaración y la comparación son las del repositorio.
"""

import json
import subprocess
from pathlib import Path

import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation import final_test_opening as opening
from mars_titan.training.learning_hold import HOLD_ENV
from tests.evaluation.test_comparison_sources import save

ROOT = Path(__file__).resolve().parents[2]
PROCEDURE = ROOT / "configs/evaluation/final-test-2024-opening.json"


def git(repository, *args):
    subprocess.run(
        ["git", "-C", str(repository), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


class Opening:
    def __init__(self, root, monkeypatch, *, allowed=True):
        self.root, self.state = root, root / "state"
        self.repository = root / "repository"
        self.repository.mkdir()
        git(self.repository, "init", "-q")
        (self.repository / "README").write_text("MARS-TITAN\n")
        git(self.repository, "add", "README")
        git(self.repository, "commit", "-q", "-m", "init")
        self.hold = root / "hold.json"
        self.allow(allowed, monkeypatch)
        self.artefacts = {}
        for name in opening.ARTEFACTS:
            path = root / "artefacts" / f"{name}.json"
            save(path, dict(name=name))
            self.artefacts[name] = path

    def allow(self, allowed, monkeypatch):
        self.hold.write_text(json.dumps(dict(training_allowed=allowed)))
        monkeypatch.setenv(HOLD_ENV, str(self.hold))

    def freeze(self, **changes):
        options = dict(
            artefacts=self.artefacts,
            trials=37,
            confirmatory=["references_vs_zero", "titans_controls"],
            repository=self.repository,
        )
        options.update(changes)
        return opening.freeze(PROCEDURE, self.state, **options)

    def check(self):
        return opening.check(PROCEDURE, self.state, repository=self.repository)

    def open(self, opener, confirm=None):
        token = self.check()["frozen_sha256"] if confirm is None else confirm
        return opening.open_final_test(
            PROCEDURE, self.state, confirm=token, repository=self.repository, opener=opener
        )

    def opener(self, frozen):
        path = self.root / "outputs" / "report.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"{frozen['commit']}\n")
        return {"report": path}

    @property
    def records(self):
        return self.state / "final-test-2024-opening.json", (
            self.repository / "reports/evaluation/final-test-2024/opening.json"
        )


@pytest.fixture
def ready(tmp_path, monkeypatch):
    study = Opening(tmp_path, monkeypatch)
    study.token = study.freeze()
    return study


def failed(study):
    return [item["condition"] for item in study.check()["conditions"] if not item["satisfied"]]


def test_the_declaration_fixes_order_conditions_and_an_unconnected_opener():
    procedure = opening.load_procedure(PROCEDURE)
    assert procedure["steps"][:4] == ["freeze", "check", "confirm", "record_opening"]
    assert procedure["steps"][-1] == "publish_record"
    assert procedure["conditions"] == list(opening.CONDITIONS)
    assert procedure["final_test"] == {"start": "2024-01-01", "end": "2025-01-01"}
    assert procedure["opener"] == "not_connected"
    assert procedure["comparison"]["name"] == "historical-masked-2000"


def test_freezing_records_hashes_trials_families_and_commit(ready):
    frozen = json.loads((ready.state / "frozen.json").read_text())
    assert ready.token == sha256(ready.state / "frozen.json")
    assert frozen["trials"] == 37 and frozen["final_test_opened"] is False
    assert frozen["confirmatory_families"] == ["references_vs_zero", "titans_controls"]
    assert "levels" in frozen["exploratory_families"]
    assert "references_vs_zero" not in frozen["exploratory_families"]
    assert frozen["artefacts"]["exclusions"]["sha256"] == sha256(ready.artefacts["exclusions"])
    head = subprocess.run(
        ["git", "-C", str(ready.repository), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert frozen["commit"] == head
    status = ready.check()
    assert status["ready"] is True and status["frozen_sha256"] == ready.token
    assert [item["condition"] for item in status["conditions"]] == list(opening.CONDITIONS)


def test_each_condition_fails_on_its_own_cause(ready, monkeypatch):
    save(ready.artefacts["trial_registry"], dict(name="otro ensayo"))
    assert failed(ready) == ["frozen_manifest_unchanged"]
    save(ready.artefacts["trial_registry"], dict(name="trial_registry"))
    (ready.repository / "notes").write_text("sin versionar\n")
    assert failed(ready) == ["repository_clean_at_frozen_commit"]
    git(ready.repository, "add", "notes")
    git(ready.repository, "commit", "-q", "-m", "later")
    assert failed(ready) == ["repository_clean_at_frozen_commit"]
    git(ready.repository, "reset", "-q", "--hard", "HEAD~1")
    ready.allow(False, monkeypatch)
    assert failed(ready) == ["learning_hold_lifted"]
    ready.allow(True, monkeypatch)
    assert failed(ready) == []


def test_without_a_connected_opener_nothing_is_recorded(ready):
    with pytest.raises(ValueError, match="no está conectado"):
        ready.open(None)
    with pytest.raises(ValueError, match="confirmación no coincide"):
        ready.open(ready.opener, confirm="0" * 64)
    assert not any(path.exists() for path in ready.records)
    with pytest.raises(ValueError, match="no está conectado"):
        opening.main(
            [
                "open",
                "--procedure",
                str(PROCEDURE),
                "--state",
                str(ready.state),
                "--repository",
                str(ready.repository),
                "--confirm",
                ready.token,
            ]
        )
    assert not any(path.exists() for path in ready.records)


def test_conditions_are_required_before_the_confirmation(ready, monkeypatch):
    ready.allow(False, monkeypatch)
    with pytest.raises(ValueError, match="learning_hold_lifted"):
        ready.open(ready.opener, confirm=ready.token)
    assert not any(path.exists() for path in ready.records)


def test_a_successful_opening_is_recorded_twice_and_cannot_repeat(ready):
    record = ready.open(ready.opener)
    ledger, published = ready.records
    assert record["status"] == "opened" and record["frozen_sha256"] == ready.token
    assert json.loads(ledger.read_text()) == json.loads(published.read_text()) == record
    assert record["outputs"]["report"]["sha256"] == sha256(ready.root / "outputs/report.txt")
    assert failed(ready) == ["repository_clean_at_frozen_commit", "no_previous_opening"]
    with pytest.raises(ValueError, match="no_previous_opening"):
        ready.open(ready.opener, confirm=ready.token)
    with pytest.raises(ValueError, match="ya se abrió"):
        ready.freeze()
    # Borrar uno de los dos registros no habilita otra apertura.
    ledger.unlink()
    assert "no_previous_opening" in failed(ready)
    ledger.write_text("{}")
    published.unlink()
    assert "no_previous_opening" in failed(ready)


def test_a_failed_opening_also_consumes_the_only_opening(ready):
    def broken(frozen):
        raise RuntimeError("fallo al preparar")

    with pytest.raises(RuntimeError, match="fallo al preparar"):
        ready.open(broken)
    ledger, published = ready.records
    record = json.loads(ledger.read_text())
    assert record["status"] == "failed" and "fallo al preparar" in record["error"]
    assert json.loads(published.read_text()) == record
    with pytest.raises(ValueError, match="no_previous_opening"):
        ready.open(ready.opener, confirm=ready.token)


def test_a_ledger_created_meanwhile_stops_the_exclusive_creation(ready, monkeypatch):
    """Si otro proceso registra entre la comprobación y la creación, esta no se repite."""
    original = opening._state

    def racing(*args):
        result = original(*args)
        ready.records[0].write_text("{}")
        return result

    monkeypatch.setattr(opening, "_state", racing)
    with pytest.raises(FileExistsError):
        ready.open(ready.opener, confirm=ready.token)
    assert not (ready.root / "outputs").exists()


@pytest.mark.parametrize(
    "changes, message",
    [
        (dict(trials=0), "ensayos"),
        (dict(confirmatory=["not_a_family"]), "confirmatorias"),
        (dict(confirmatory=[]), "confirmatorias"),
        (dict(artefacts={}), "exactamente"),
    ],
)
def test_freezing_requires_the_declared_contents(tmp_path, monkeypatch, changes, message):
    study = Opening(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=message):
        study.freeze(**changes)


def test_freezing_requires_a_clean_repository(tmp_path, monkeypatch):
    study = Opening(tmp_path, monkeypatch)
    (study.repository / "draft").write_text("x\n")
    with pytest.raises(ValueError, match="limpio"):
        study.freeze()


@pytest.mark.parametrize(
    "change",
    [
        dict(opener="connected"),
        dict(repeat="once_per_freeze"),
        dict(conditions=list(reversed(opening.CONDITIONS))),
        dict(final_test={"start": "2023-01-01", "end": "2025-01-01"}),
        dict(repository_record="../outside.json"),
        dict(ledger="nested/record.json"),
    ],
)
def test_the_declaration_cannot_be_relaxed(tmp_path, change):
    declared = json.loads(PROCEDURE.read_text())
    path = tmp_path / "procedure.json"
    declared["comparison"] = str(PROCEDURE.parent / declared["comparison"])
    save(path, dict(declared, **change))
    with pytest.raises(ValueError, match="no cumple su contrato"):
        opening.load_procedure(path)


def test_a_comparison_changed_after_freezing_invalidates_the_manifest(ready):
    path = ready.state / "frozen.json"
    frozen = json.loads(path.read_text())
    frozen["comparison"]["protocol_sha256"]["US"]["US"] = "0" * 64
    save(path, frozen)
    status = ready.check()
    assert failed(ready) == ["frozen_manifest_unchanged"]
    assert "comparación" in status["conditions"][0]["detail"]
