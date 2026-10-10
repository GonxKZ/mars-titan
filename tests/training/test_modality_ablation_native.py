"""Ablación de modalidades en las familias con banco episódico nativo, en CPU y sin ajustes.

MARS-TITAN y la GRU candidata necesitan el enlace episódico compilado. Cada familia se
ajusta dos veces sobre el mismo corpus técnico, con las modalidades presentes y con
noticias y fundamentales ausentes de verdad, con optimizadores que solo registran
llamadas. Como los parámetros siguen siendo los iniciales de la semilla, el traslado
enmascarado del primer corpus debe dar las mismas predicciones que el del segundo, donde
la ablación no cambia nada.
"""

import os
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.modality_ablation import ablation_identity
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import candidate_walk_forward as candidate
from mars_titan.training import mars_titan_walk_forward as mw
from mars_titan.training.learning_hold import HOLD_ENV
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_financial_run import RecordingOptimizer
from tests.training.test_modality_ablation_carry import (
    permitted,  # noqa: F401
    same_rows,
    titans,  # noqa: F401
)
from tests.training.test_modality_ablation_inputs import pattern, without
from tests.training.test_titans_walk_forward import unfused_attention
from tests.training.test_walk_forward_v2_views import PROTOCOLS, prepare, sessions

pytestmark = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)
BOTH = "mask_news_and_fundamentals"


def evaluation(folder, receipt):
    return pq.read_table(folder / receipt["predictions"]["evaluation"]["path"])


def assert_same_predictions(left, right):
    left, right = same_rows(left, right)
    for name in ("prediction", *QUANTILE_COLUMNS):
        np.testing.assert_array_equal(left[name].to_numpy(), right[name].to_numpy())


@pytest.fixture(scope="module")
def readers(titans, tmp_path_factory):  # noqa: F811
    """Lectores M1 de MARS-TITAN sobre las ventanas Titans-MAC de los dos corpus."""
    from tests.training.test_mars_titan_walk_forward import M1, mars, readout_recipe

    root = tmp_path_factory.mktemp("mars-ablation")
    runs = {}
    with pytest.MonkeyPatch.context() as patch, unfused_attention():
        patch.setenv(HOLD_ENV, str(titans.hold))
        plan = readout_recipe(root)
        for name in ("original", "absent"):
            run = getattr(titans, name)
            mars(run.view, run.output, plan, root / name, M1)
            runs[name] = SimpleNamespace(view=run.view, output=root / name)
    return SimpleNamespace(**runs)


def test_mars_titan_ablation_equals_the_reader_with_a_real_absence(
    readers,
    tmp_path,
    permitted,  # noqa: F811
):
    """El banco episódico y la memoria del padre ven la ausencia desde el calentamiento."""
    carried = {}
    for name in ("original", "absent"):
        run = getattr(readers, name)
        with unfused_attention():
            receipt = mw.carry_mars_titan(
                run.output,
                run.view,
                run.view,
                tmp_path / name,
                device="cpu",
                modality_ablation=BOTH,
            )
        assert set(receipt["predictions"]) == {"evaluation"}
        assert receipt["modality_ablation"] == ablation_identity(BOTH)
        carried[name] = evaluation(tmp_path / name, receipt)
    assert_same_predictions(carried["original"], carried["absent"])
    fitted = pq.read_table(readers.original.output / "evaluation-predictions.parquet")
    fitted, masked = same_rows(fitted, carried["original"])
    assert (fitted["prediction"].to_numpy() != masked["prediction"].to_numpy()).any()


@pytest.fixture(scope="module")
def episodic(tmp_path_factory):
    """Ventanas de la GRU candidata ajustadas con un optimizador que no modifica pesos."""
    from tests.training.test_candidate_walk_forward import MODEL, SMALL

    root = tmp_path_factory.mktemp("candidate-ablation")
    hold = root / "training-hold.json"
    hold.write_text('{"training_allowed": true}', encoding="utf-8")
    runs = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOLD_ENV, str(hold))
        for name, presence in (("original", pattern), ("absent", without(BOTH))):
            data = historical_temporal_fixture(
                root / name / "data", ("US",), {"US": sessions("US")}, assets=2, presence=presence
            )
            prepare(data, PROTOCOLS["US"], root / name / "views")
            view = root / name / "views" / "fold-000" / "manifest.json"
            candidate.fit_window(
                view,
                root / name / "anchor",
                SMALL,
                seed=42,
                model=MODEL,
                parent_id="US/fold-000/gru_episodic",
                device="cpu",
                warmup_months=12,
                optimizer_factory=RecordingOptimizer,
            )
            runs[name] = SimpleNamespace(view=view, output=root / name / "anchor")
    return SimpleNamespace(hold=hold, **runs)


def test_candidate_ablation_equals_the_window_with_a_real_absence(episodic, tmp_path, monkeypatch):
    monkeypatch.setenv(HOLD_ENV, str(episodic.hold))
    carried = {}
    for name in ("original", "absent"):
        run = getattr(episodic, name)
        receipt = candidate.carry_window(
            run.output,
            run.view,
            run.view,
            tmp_path / name,
            parent_id="US/fold-000/gru_episodic",
            device="cpu",
            modality_ablation=BOTH,
        )
        assert set(receipt["predictions"]) == {"evaluation"}
        assert receipt["modality_ablation"] == ablation_identity(BOTH)
        # Una predicción ablacionada no publica recibos walk-forward.
        assert receipt["receipts"] == {}
        carried[name] = pq.read_table(
            tmp_path / name / receipt["predictions"]["evaluation"]["path"]
        )
    assert_same_predictions(carried["original"], carried["absent"])
