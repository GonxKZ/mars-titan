"""Presupuesto de entrada explícito, limitado al ejecutor de referencias."""

import importlib

import pytest

from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_corpus_inputs import corpus

VARIABLE = "MARS_TITAN_INPUT_CACHE_MIB"


def configured(path):
    return importlib.import_module("mars_titan.training.reference_run").configured_corpus(path)


def test_reference_cache_default_preserves_arrays_only_and_one_gib(tmp_path, monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    reader = configured(corpus(tmp_path))
    assert reader.cache_limit == 1024**3
    assert reader.cache_sample_tables is False


@pytest.mark.parametrize("mib", [1, 1024, 2048, 4096])
def test_reference_explicit_cache_budget_enables_sample_tables(tmp_path, monkeypatch, mib):
    monkeypatch.setenv(VARIABLE, str(mib))
    reader = configured(corpus(tmp_path))
    assert reader.cache_limit == mib * 1024**2
    assert reader.cache_sample_tables is True


@pytest.mark.parametrize(
    "setting", ["", "0", "4097", "-1", "1.5", "true", " 2048 ", "١", "9" * 1000]
)
def test_invalid_cache_budget_is_rejected_before_reading_manifest(tmp_path, monkeypatch, setting):
    monkeypatch.setenv(VARIABLE, setting)
    with pytest.raises(ValueError, match="MARS_TITAN_INPUT_CACHE_MIB"):
        configured(tmp_path / "missing.json")


def test_direct_reader_ignores_reference_runner_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(VARIABLE, "2048")
    reader = CorpusDataset(corpus(tmp_path))
    assert reader.cache_limit == 1024**3
    assert reader.cache_sample_tables is False
