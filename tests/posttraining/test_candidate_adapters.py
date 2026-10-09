"""Cabeza adaptada y continuación de la GRU candidata nativa, sin pasos que ajusten pesos.

Los padres son ventanas walk-forward de la candidata ajustadas con el registrador, que no
cambia pesos, sobre la vista conjunta US+CN del corpus técnico. Necesitan el enlace nativo.
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.models.predictive_adaptation import adapter_names
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.models.titans.financial_inputs import validated_cpu_batch
from mars_titan.posttraining import candidate_adapters as ca
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.posttraining.adapter_matrix import read_matrix
from mars_titan.training import candidate_walk_forward as walk
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.training.test_candidate_walk_forward import allowed as allowed
from tests.training.test_candidate_walk_forward import fit
from tests.training.test_candidate_walk_forward import views as views
from tests.training.test_financial_run import RecordingOptimizer

pytestmark = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)

ROOT = Path(__file__).resolve().parents[2]
SCOPE, FIRST, NEXT = "US+CN", "fold-000", "fold-001"
COLUMNS = ["asset_id", "prediction_at", "target", "prediction", *QUANTILE_COLUMNS]


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    """Matriz v3 con una semilla y una época, para recorrer la ventana técnica."""
    document = json.loads((ROOT / "configs/posttraining/adapter-matrix-v3.json").read_text())
    document["budget"].update(seeds=[42], epochs=1)
    path = tmp_path_factory.mktemp("matrix") / "matrix.json"
    path.write_text(json.dumps(document))
    return read_matrix(path)


@pytest.fixture(scope="module")
def parent(views, allowed):
    output = views.root / "adapter-parent"
    report, _ = fit(
        views.windows[SCOPE][FIRST],
        output,
        factory=RecordingOptimizer,
        parent=f"{SCOPE}/{FIRST}/gru_episodic",
    )
    assert report["status"] == "completed" and report["markets"] == ["CN", "US"]
    return output


def cases(matrix):
    document, digest = matrix
    return {
        row["id"].split("/", 1)[1]: row["case"]
        for row in cm.cases(document, digest, "episodic_gru")
    }


@pytest.fixture(scope="module")
def arms(views, parent, matrix, allowed):
    document, digest = matrix
    result = {}
    for name, case in cases(matrix).items():
        output = views.root / f"adapter-{name}"
        made = []

        def factory(groups, made=made):
            made.append(RecordingOptimizer(groups))
            return made[-1]

        report = ca.run_candidate_posttraining(
            parent,
            views.windows[SCOPE][FIRST],
            output,
            case=case,
            matrix=document,
            digest=digest,
            device="cpu",
            optimizer_factory=factory,
        )
        result[name] = (output, report, made[0])
    return result


def rows(folder, record):
    return pq.read_table(Path(folder) / record["path"], columns=COLUMNS).sort_by(
        [("asset_id", "ascending"), ("prediction_at", "ascending")]
    )


def test_python_head_reproduces_the_native_head_bit_for_bit(views):
    from mars_titan.models.candidate.input_adapter import CandidateInputAdapter

    source = _source(views)
    specification = source.specification()
    raw = next(event for event in source.batched_events(block_rows=7) if event.inputs).inputs[0]
    for dtype in (torch.float64, torch.float32):
        adapter = CandidateInputAdapter(specification, dtype=dtype, parameter_seed=42)
        prediction = adapter.forward(validated_cpu_batch(raw, specification))
        state = prediction.native.state
        assert torch.equal(ca.CandidateHead(adapter.model)(state), prediction.quantiles)
        adapted = ca.adapted_head(adapter.model, {"head": {"form": "residual"}}, seed=3)
        assert torch.equal(adapted(state), prediction.quantiles)
        assert adapter_names(adapted) == [
            "parametrizations.weight.0.delta",
            "parametrizations.bias.0.delta",
        ]


def _source(views):
    """Tramo de ajuste de la ventana conjunta, con lotes de un solo corte."""
    sources = walk.window_sources(
        CorpusDataset(views.windows[SCOPE][FIRST], input_policy=HISTORICAL_MASKED),
        views.root / "specification-indices",
        ("train",),
    )
    return sources["train"]


def test_head_arm_emits_the_parent_rows_of_both_markets(arms, parent):
    parent_report = json.loads((parent / "window.json").read_text())
    output, report, _ = arms["head"]
    assert report["status"] == "completed" and report["markets"] == ["CN", "US"]
    for name in ("validation", "calibration", "evaluation"):
        left = rows(output, report["predictions"][name])
        right = rows(parent, parent_report["predictions"][name])
        assert left.equals(right), name
        markets = pq.read_table(output / report["predictions"][name]["path"])["market"]
        assert set(markets.to_pylist()) == {"CN", "US"}


def test_arms_share_updates_and_only_the_head_corrections_move(arms):
    _, head, head_optimizer = arms["head"]
    _, full, full_optimizer = arms["full_continuation"]
    assert head["global_step"] == full["global_step"] == head_optimizer.calls > 0
    assert [group["role"] for group in head_optimizer.param_groups] == ["head_adapters"]
    weight, bias = head_optimizer.param_groups[0]["params"]
    assert weight.shape[0] == bias.shape[0] == 5
    posttraining = head["posttraining"]
    assert posttraining["adapter"]["trainable_parameters"] == weight.numel() + bias.numel()
    assert posttraining["native_parameters_sha256"] == head["checkpoint"]["parameters_sha256"]
    first = head_optimizer.records[0]
    assert all(value is not None and value.abs().sum() > 0 for value in first.values())
    assert {group["role"] for group in full_optimizer.param_groups} >= {"encoder", "head"}
    assert full["posttraining"]["native_parameters_sha256"] is None


@pytest.fixture(scope="module")
def frozen(views, parent, allowed):
    output = views.root / "adapter-frozen"
    receipt = ca.frozen_candidate(
        parent, views.windows[SCOPE][FIRST], views.windows[SCOPE][NEXT], output, device="cpu"
    )
    return output, receipt


def test_frozen_parent_matches_the_carried_parent(frozen, parent, views, tmp_path):
    output, receipt = frozen
    assert receipt["kind"] == ca.FROZEN_KIND and receipt["status"] == "completed"
    assert set(receipt["predictions"]) == set(ca.PREDICTED)
    assert receipt["months_since_parent_information"] > 0
    base = walk.carry_window(
        parent,
        views.windows[SCOPE][FIRST],
        views.windows[SCOPE][NEXT],
        tmp_path / "base",
        parent_id=f"{SCOPE}/{FIRST}/gru_episodic",
        device="cpu",
    )
    for name in ("calibration", "evaluation"):
        left = rows(output, receipt["predictions"][name])
        right = rows(tmp_path / "base", base["predictions"][name])
        assert left.equals(right), name
    validation = rows(output, receipt["predictions"]["validation"])
    counts = CorpusDataset(views.windows[SCOPE][NEXT], input_policy=HISTORICAL_MASKED).manifest[
        "counts"
    ]
    assert validation.num_rows == counts["validation"] > 0


def test_staged_head_fits_the_new_rows_and_reproduces_the_frozen_parent(
    views, parent, matrix, frozen, allowed, tmp_path
):
    document, digest = matrix
    made = []

    def factory(groups):
        made.append(RecordingOptimizer(groups))
        return made[-1]

    report = ca.run_candidate_posttraining(
        parent,
        views.windows[SCOPE][NEXT],
        tmp_path / "staged",
        case=cases(matrix)["head"],
        matrix=document,
        digest=digest,
        device="cpu",
        optimizer_factory=factory,
        parent_view=views.windows[SCOPE][FIRST],
    )
    assert report["status"] == "completed" and made and made[0].calls > 0
    placement = report["posttraining"]["placement"]
    assert placement["design"] == ca.STAGED and placement["parent_window"] == FIRST
    assert report["request"]["parent_view_sha256"] == placement["parent_view_sha256"]
    train = report["sources"]["train"]["phase"]
    since = int(np.datetime64(placement["fit_start"], "us").astype(np.int64))
    until = int(np.datetime64(placement["fit_end"], "us").astype(np.int64))
    # Sin calentamiento: las 64 sesiones de contexto viajan en cada muestra.
    assert (train["warmup_start"], train["decision_start"]) == (since, since)
    assert (train["decision_end"], train["close_at"]) == (until, until)
    # Con la corrección de la cabeza a cero y sin cambios de pesos, emite al padre congelado.
    output, receipt = frozen
    for name in ca.PREDICTED:
        left = rows(tmp_path / "staged", report["predictions"][name])
        right = rows(output, receipt["predictions"][name])
        assert left.equals(right), name


def test_cases_from_another_matrix_or_seed_are_rejected(views, parent, matrix, tmp_path):
    document, digest = matrix
    case = cases(matrix)["head"]
    common = dict(matrix=document, device="cpu", optimizer_factory=RecordingOptimizer)
    with pytest.raises(ValueError, match="matriz"):
        ca.run_candidate_posttraining(
            parent,
            views.windows[SCOPE][FIRST],
            tmp_path / "a",
            case=case,
            digest="0" * 64,
            **common,
        )
    with pytest.raises(ValueError, match="semilla"):
        ca.run_candidate_posttraining(
            parent,
            views.windows[SCOPE][FIRST],
            tmp_path / "b",
            case=dict(case, seed=43),
            digest=digest,
            **common,
        )
