"""El medidor solo carga la referencia declarada y no sustituye recibos anteriores."""

import sys
from types import SimpleNamespace

import pytest

from benchmarks import neutral_abstention_review as review


def test_wrong_reference_hash_is_rejected_before_import(tmp_path):
    source = tmp_path / "reference.py"
    source.write_text('raise RuntimeError("Esta fuente no debe ejecutarse")')
    with pytest.raises(ValueError, match="huella"):
        review.load_reference(source)


def test_reference_must_be_a_regular_file(tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(ValueError, match="regular"):
        review.load_reference(missing)
    link = tmp_path / "link.py"
    link.symlink_to(missing)
    with pytest.raises(ValueError, match="regular"):
        review.load_reference(link)


def test_existing_receipt_is_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    output.write_text("original")
    monkeypatch.setattr(
        sys, "argv", ["review", "--reference", str(tmp_path / "absent.py"), "--output", str(output)]
    )
    with pytest.raises(FileExistsError):
        review.main()
    assert output.read_text() == "original"


@pytest.mark.parametrize("repeats", ["0", "21"])
def test_cli_rejects_repetition_counts_outside_its_budget(repeats, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["review", "--reference", "absent.py", "--output", "unused.json", "--repeats", repeats],
    )
    with pytest.raises(SystemExit) as error:
        review.main()
    assert error.value.code == 2


@pytest.mark.parametrize("changed", ["volatility", "outputs"])
def test_measurement_rejects_a_numerical_regression(changed, monkeypatch):
    reference = SimpleNamespace(
        **{
            name: getattr(review.corrected, name)
            for name in ("fit", "past_volatility", "seed_disagreement", "emit")
        }
    )
    if changed == "volatility":
        original = reference.past_volatility
        monkeypatch.setattr(
            review.corrected, "past_volatility", lambda *a, **kw: original(*a, **kw) + 1.0
        )
    else:
        original = reference.emit

        def altered(*args, **kwargs):
            outputs, summary = original(*args, **kwargs)
            return {k: values + 1.0 for k, values in outputs.items()}, summary

        monkeypatch.setattr(review.corrected, "emit", altered)
    with pytest.raises(AssertionError):
        review.measure(reference, 16, 1, repeats=1)
