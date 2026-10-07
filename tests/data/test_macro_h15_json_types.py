"""Los tipos JSON del contrato no se sustituyen por valores numéricamente iguales."""

import json

import pytest

from mars_titan.data.storage import sha256
from tests.data import test_macro_h15_contexts as fixtures
from tests.data.test_macro_h15_contexts import load, produce


@pytest.fixture
def source(tmp_path):
    return fixtures.source.__wrapped__(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_ready", 0),
        ("admission_required", 1),
        ("final_test_opened", 0),
        ("reused", 0),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("observations", 30.0),
    ],
)
def test_receipt_rejects_equal_boolean_and_numeric_substitutions(source, tmp_path, field, value):
    load(source)
    output = tmp_path / "panel"
    produce(source, output)
    path = output / "report.json"
    report = json.loads(path.read_text())
    assert report[field] == value and type(report[field]) is not type(value)
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="recibo"):
        produce(source, output)


@pytest.mark.parametrize(
    "keys",
    [
        ("schema_version",),
        ("calculation", "daily_lag_policy_version"),
        ("decisions", "US"),
        ("limits", "cells"),
        ("limits", "artifact_bytes"),
    ],
)
def test_configuration_preserves_numeric_types_even_with_updated_hash(source, tmp_path, keys):
    load(source)
    output = tmp_path / "panel"
    produce(source, output)
    path = output / "configuration.json"
    config = json.loads(path.read_text())
    selected = config
    for key in keys[:-1]:
        selected = selected[key]
    value = selected[keys[-1]]
    selected[keys[-1]] = True if value == 1 else float(value)
    path.write_text(json.dumps(config))
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report["configuration_sha256"] = sha256(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="recibo"):
        produce(source, output)


@pytest.mark.parametrize(
    "keys",
    [
        ("observed",),
        ("rows",),
        ("decisions",),
        ("missing_reasons", "source_unavailable:not_in_h15_document_bundle"),
    ],
)
def test_recovery_requires_integer_coverage_counts(source, tmp_path, keys):
    load(source)
    output = tmp_path / "panel"
    produce(source, output)
    path = output / "report.json"
    report = json.loads(path.read_text())
    selected = report["markets"]["US"]
    for key in keys[:-1]:
        selected = selected[key]
    value = selected[keys[-1]]
    selected[keys[-1]] = False if value == 0 else float(value)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="recuentos|recibo"):
        produce(source, output)


@pytest.mark.parametrize("replacement", [False, 0])
def test_documentary_zero_keeps_its_numeric_type_in_events(source, tmp_path, replacement):
    load(source)
    output = tmp_path / "panel"
    produce(source, output)
    path = output / "events.json"
    events = json.loads(path.read_text())
    zero = next(row for row in events if row["document_value"] == 0.0)
    assert type(zero["document_value"]) is float
    zero["document_value"] = replacement
    path.write_text(json.dumps(events))
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report["artifacts"][path.name] = sha256(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="eventos"):
        produce(source, output)


def test_valid_json_types_recover_without_changing_files(source, tmp_path):
    load(source)
    output = tmp_path / "panel"
    initial = produce(source, output)
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in output.iterdir()}
    recovered = produce(source, output)
    assert recovered == dict(initial, reused=True)
    assert type(recovered["schema_version"]) is int
    assert type(recovered["training_ready"]) is bool
    assert type(recovered["admission_required"]) is bool
    assert type(recovered["observations"]) is int
    assert type(recovered["markets"]["US"]["observed"]) is int
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in before}
