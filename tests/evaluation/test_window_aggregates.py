"""Agregados por ventana de la comparación: relectura bit a bit e informe sin filas.

Se usa el estudio sintético de la comparación con estratos y ablación. Las pruebas
comprueban que el informe calculado desde los agregados es idéntico al que lee las
predicciones, también después de liberar todas las filas, y que unos agregados no se
aceptan con otras fuentes, otra configuración u otro código.
"""

import collections
import copy
import dataclasses
import json
import struct

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.data import prediction_files
from mars_titan.data.storage import sha256
from mars_titan.evaluation import modality_strata as strata
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation import window_aggregates as aggregates
from mars_titan.evaluation.forecast_scores import SessionScores
from tests.evaluation.test_comparison_sources import save
from tests.evaluation.test_modality_ablation import VOLATILE, declare, write_ablation
from tests.evaluation.test_modality_strata import by_asset, provider
from tests.evaluation.test_walk_forward_comparison import Study


def primary(report):
    return {key: value for key, value in report.items() if key not in VOLATILE}


def loaded(study, ablation_path=None):
    config = walk.load_config(study.config_path)
    sources = walk.load_sources(study.sources_path, config, study.scope)
    ablation = None
    if ablation_path is not None:
        ablation = walk._ablation_sources(ablation_path, config, sources)
    return config, sources, ablation


def prediction_records(study, ablation_path):
    """Todas las predicciones del estudio con su huella, también las enmascaradas."""
    records = []
    for arm in study.sources["arms"].values():
        for windows in arm.values():
            for entry in windows.values():
                for part in ("calibration", "evaluation"):
                    if part in entry:
                        path = study.sources_path.parent / entry[part]["path"]
                        records.append((path, entry[part]["sha256"]))
    document = json.loads(ablation_path.read_text())
    for arms in document["variants"].values():
        for seeds in arms.values():
            for windows in seeds.values():
                for entry in windows.values():
                    records.append(
                        (
                            ablation_path.parent / entry["evaluation"]["path"],
                            entry["evaluation"]["sha256"],
                        )
                    )
    return records


@pytest.fixture(scope="module")
def released(tmp_path_factory):
    """El informe leído de las filas, los agregados de cada ventana y el informe sin filas."""
    root = tmp_path_factory.mktemp("aggregates")
    with pytest.MonkeyPatch.context() as patch:
        study = declare(Study(root), version=3)
        patch.setattr(strata, "view_presence", provider(study, by_asset))
        ablation_path = write_ablation(study)
        expected = walk.evaluate_walk_forward(
            study.config_path, study.sources_path, study.scope, ablation_sources=ablation_path
        )
        config, sources, ablation = loaded(study, ablation_path)
        folder = root / "aggregates"
        written = {
            window: aggregates.write(folder, config, sources, window, ablation)
            for window in sources["windows"]
        }
        from_aggregates = walk.evaluate_walk_forward(
            study.config_path,
            study.sources_path,
            study.scope,
            ablation_sources=ablation_path,
            aggregates=folder,
        )
        for path, digest in prediction_records(study, ablation_path):
            prediction_files.release(path, digest, stage="fixture")
        after = walk.evaluate_walk_forward(
            study.config_path,
            study.sources_path,
            study.scope,
            ablation_sources=ablation_path,
            aggregates=folder,
        )
        with pytest.raises(prediction_files.PredictionsReleased):
            walk.evaluate_walk_forward(
                study.config_path, study.sources_path, study.scope, ablation_sources=ablation_path
            )
        yield dict(
            study=study,
            ablation=ablation_path,
            folder=folder,
            written=written,
            expected=expected,
            from_aggregates=from_aggregates,
            after=after,
        )


def test_the_report_from_window_aggregates_is_identical_to_reading_the_rows(released):
    expected_report, expected_sessions = released["expected"]
    for report, sessions in (released["from_aggregates"], released["after"]):
        assert primary(report) == primary(expected_report)
        assert sessions.equals(expected_sessions)
    assert "modality_ablation" in expected_report and "modality_strata" in expected_report
    assert expected_report["modality_ablation"]["status"] != "pending"


def test_window_aggregates_are_small_files_with_their_digest(released):
    study = released["study"]
    config, sources, ablation = loaded(study, released["ablation"])
    for window, record in released["written"].items():
        results, _ = aggregates.read(released["folder"], config, sources, window, ablation)
        scores = results["gru", 42]["raw"]
        # Las puntuaciones por sesión son float64 y no se redondean al guardarlas.
        assert scores.mae.dtype == scores.pinball.dtype == np.float64
        assert record["path"].name == f"{window}.npz"
        assert record["sha256"] == sha256(record["path"])
        assert 0 < record["bytes"] == record["path"].stat().st_size


def test_aggregates_reject_other_predictions_configuration_or_code(released, tmp_path, monkeypatch):
    study = released["study"]
    monkeypatch.setattr(strata, "view_presence", provider(study, by_asset))
    config, sources, ablation = loaded(study, released["ablation"])
    window = next(iter(sources["windows"]))
    folder = released["folder"]
    assert aggregates.same(
        aggregates.read(folder, config, sources, window, ablation),
        aggregates.read(folder, config, sources, window, ablation),
    )
    other = copy.deepcopy(sources)
    key = next(k for k in other["files"] if k[2] == window)
    other["files"][key]["evaluation"] = dict(other["files"][key]["evaluation"], sha256="0" * 64)
    with pytest.raises(ValueError, match="predictions"):
        aggregates.read(folder, config, other, window, ablation)
    with pytest.raises(ValueError, match="comparison_sha256"):
        aggregates.read(folder, dict(config, sha256="1" * 64), sources, window, ablation)
    with pytest.raises(ValueError, match="ablation"):
        aggregates.read(folder, config, sources, window, None)
    original = aggregates.sha256

    def changed(path):
        return "2" * 64 if path.name == "forecast_scores.py" else original(path)

    monkeypatch.setattr(aggregates, "sha256", changed)
    with pytest.raises(ValueError, match="code"):
        aggregates.read(folder, config, sources, window, ablation)
    monkeypatch.setattr(aggregates, "sha256", original)
    with pytest.raises(ValueError, match="Faltan los agregados"):
        aggregates.read(tmp_path, config, sources, window, ablation)


def test_one_window_manifest_validates_with_the_restricted_configuration(released, tmp_path):
    study = released["study"]
    config = walk.load_config(study.config_path)
    window = study.folds[-1]["id"]
    manifest = copy.deepcopy(study.sources)
    manifest["windows"] = {window: manifest["windows"][window]}
    for seeds in manifest["arms"].values():
        for entries in seeds.values():
            for name in list(entries):
                if name != window:
                    entries.pop(name)
                    continue
                for part in entries[name].values():
                    if isinstance(part, dict) and "path" in part:
                        part["path"] = str(study.sources_path.parent / part["path"])
    save(tmp_path / "one.json", manifest)
    with pytest.raises(ValueError, match="exactamente las ventanas"):
        walk.load_sources(tmp_path / "one.json", config, study.scope)
    restricted = walk.restrict_windows(config, study.scope, [window])
    assert restricted["sha256"] == config["sha256"]
    one = walk.load_sources(tmp_path / "one.json", restricted, study.scope)
    full = walk.load_sources(study.sources_path, config, study.scope)
    assert list(one["windows"]) == [window]
    assert aggregates.identity(restricted, one, window) == aggregates.identity(config, full, window)
    with pytest.raises(ValueError, match="ventanas deben ser"):
        walk.restrict_windows(config, study.scope, ["fold-999"])
    with pytest.raises(ValueError, match="ventanas deben ser"):
        walk.restrict_windows(config, study.scope, [])


@dataclasses.dataclass(frozen=True)
class Foreign:
    value: int


def test_encoding_keeps_types_bits_and_nested_keys():
    nan = struct.unpack("<d", bytes.fromhex("230100000000f87f"))[0]
    scores = object.__new__(SessionScores)
    for field in dataclasses.fields(SessionScores):
        object.__setattr__(scores, field.name, None)
    object.__setattr__(scores, "markets", ("US", "CN"))
    object.__setattr__(scores, "mae", np.array([0.5, -0.0, np.nan], dtype=np.float64))
    object.__setattr__(scores, "rank_ic_status", np.array(["ok", "few"]))
    value = {
        ("gru", 42): dict(raw=scores, calibrated=None, score=-0.0, nan=nan),
        ("zero", None): [1, True, "x", np.float64(0.25), np.int64(-3), (1.5, [2])],
    }
    arrays = {}
    skeleton = aggregates.encode(value, arrays)
    back = aggregates.decode(json.loads(json.dumps(skeleton)), arrays)
    assert aggregates.same(back, value)
    assert struct.pack("<d", back["gru", 42]["nan"]) == struct.pack("<d", nan)
    assert struct.pack("<d", back["gru", 42]["score"]) == struct.pack("<d", -0.0)
    assert type(back["zero", None][5]) is tuple and type(back["zero", None][3]) is np.float64
    assert isinstance(back["gru", 42]["raw"], SessionScores)
    assert not aggregates.same(back, {**value, ("zero", None): [1, True, "x"]})
    assert not aggregates.same(np.array([0.0]), np.array([-0.0]))
    assert not aggregates.same(np.array([1.0]), np.array([1], dtype=np.int64))
    assert not aggregates.same((1,), [1])
    assert not aggregates.same(0.0, -0.0)


@pytest.mark.parametrize(
    "value",
    [
        np.array([object()], dtype=object),
        np.array([0.5], dtype=np.float32),
        np.array([0.5], dtype=np.float16),
        np.float32(0.5),
        collections.Counter(a=1),
        Foreign(1),
        {1, 2},
        pa.array([1]),
    ],
)
def test_encoding_rejects_values_it_cannot_restore_exactly(value):
    with pytest.raises(ValueError):
        aggregates.encode(value, {})


def test_decoding_only_rebuilds_project_dataclasses_with_their_fields():
    with pytest.raises(ValueError, match="No se reconstruye"):
        aggregates.decode({"c": "os:PathLike", "v": {}}, {})
    with pytest.raises(ValueError, match="cambió sus campos"):
        aggregates.decode(
            {"c": "mars_titan.evaluation.forecast_scores:SessionScores", "v": {"mae": None}}, {}
        )
    with pytest.raises(ValueError, match="no es de datos"):
        aggregates.decode(
            {"c": "mars_titan.evaluation.forecast_scores:score_sessions", "v": {}}, {}
        )


def test_written_aggregates_are_rejected_when_their_file_is_not_aggregates(released, tmp_path):
    study = released["study"]
    config = walk.load_config(study.config_path)
    sources = walk.load_sources(study.sources_path, config, study.scope)
    window = next(iter(sources["windows"]))
    path = aggregates.path_for(tmp_path, study.scope, window)
    path.parent.mkdir(parents=True)
    skeleton = json.dumps(dict(kind="other", schema_version=1)).encode()
    np.savez(path, __skeleton__=np.frombuffer(skeleton, dtype=np.uint8))
    with pytest.raises(ValueError, match="no son agregados"):
        aggregates.read(tmp_path, config, sources, window)
