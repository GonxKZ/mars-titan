"""Traslados con ablación de modalidades, en CPU y sin modificar ningún peso.

Las anclas son las de las pruebas de traslado: una referencia neuronal con los pesos
iniciales de su semilla, un modelo tabular que suma los bits de presencia y ventanas de
Titans-MAC ajustadas con un optimizador que solo registra gradientes. Se comprueba que la
ablación solo cambia lo que lee el modelo, que un modelo sin memoria conserva las filas
sin la modalidad y que la memoria de Titans-MAC ve la ausencia desde su calentamiento. La
corrección B6 sobre esas ventanas no tiene parámetros, así que su ablación se compara del
mismo modo con la corrección sobre el corpus sin noticias ni fundamentales.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES
from mars_titan.data.modality_ablation import VARIANTS, ablation_identity
from mars_titan.data.storage import sha256
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import carried_predictions as carry
from mars_titan.training import mars_titan_correction as mc
from mars_titan.training import mars_titan_walk_forward as mw
from mars_titan.training import modality_ablation_stage as ablation_stage
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.learning_hold import HOLD_ENV
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_carried_predictions import (
    PresenceCount,
    cpu,  # noqa: F401
    neural_anchor,
    reader,
    tabular_anchor,
)
from tests.training.test_modality_ablation_inputs import pattern
from tests.training.test_titans_walk_forward import recipe, unfused_attention, window
from tests.training.test_titans_walk_forward import views as titans_views
from tests.training.test_walk_forward_v2_views import PROTOCOLS, prepare, sessions

CORRECTION = Path("configs/titans/mature-correction-historical-masked.json")
B6 = {"associative_memory": {"rule": "proximal", "key": "codec"}}
BOTH = "mask_news_and_fundamentals"


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    """Vistas anuales de dos activos con noticias y fundamentales en filas alternas."""
    root = tmp_path_factory.mktemp("ablation-carry")
    data = historical_temporal_fixture(
        root / "data", ("US",), {"US": sessions("US")}, assets=2, presence=pattern
    )
    prepare(data, PROTOCOLS["US"], root / "views")
    return SimpleNamespace(
        root=root, view=lambda index: root / f"views/fold-{index:03d}/manifest.json"
    )


def manifest(path):
    return json.loads(path.read_text())


def masked_bits(presence, variant):
    return presence[:, [MODALITIES.index(name) for name in VARIANTS[variant]]]


def test_same_window_is_admitted_only_for_an_ablation_of_that_view(views):
    first, second = manifest(views.view(0)), manifest(views.view(1))
    anchor, target, months = carry.carried_window(
        first, first, input_policy=HISTORICAL_MASKED, same_window=True
    )
    assert anchor == target and anchor["evaluation"] == ["2005-01-01", "2006-01-01"]
    assert anchor["validation"][1] <= target["calibration"][0] and months == 3
    with pytest.raises(ValueError, match="no es posterior"):
        carry.carried_window(first, first, input_policy=HISTORICAL_MASKED)
    with pytest.raises(ValueError, match="no es posterior"):
        carry.carried_window(second, first, input_policy=HISTORICAL_MASKED, same_window=True)
    # Otra vista con la misma ventana no es la vista del ancla.
    other = json.loads(json.dumps(first))
    other["counts"]["evaluation"] += 1
    with pytest.raises(ValueError, match="no es posterior"):
        carry.carried_window(first, other, input_policy=HISTORICAL_MASKED, same_window=True)
    assert carry.predicted_partitions(None) == ("calibration", "evaluation")
    assert carry.predicted_partitions("mask_news") == ("evaluation",)
    # El padre congelado de la cadena predice además la validación de la ventana posterior.
    assert carry.predicted_partitions(None, frozen_parent=True) == (
        "validation",
        "calibration",
        "evaluation",
    )
    for options in (dict(modality_ablation="mask_news"), dict(regenerate=True)):
        with pytest.raises(ValueError, match="sin ablación ni regeneración"):
            carry.predicted_partitions(
                options.get("modality_ablation"), options.get("regenerate", False), True
            )
    assert carry.frozen_parent_record(False) == {}
    assert carry.frozen_parent_record(True) == dict(frozen_parent=True)
    assert carry.ablation_record(None) == {}
    assert carry.ablation_record("mask_news") == dict(
        modality_ablation=ablation_identity("mask_news")
    )


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_tabular_ablation_subtracts_exactly_the_masked_presence(
    views, tmp_path, monkeypatch, variant
):
    monkeypatch.setattr(carry, "_tabular_model", lambda *args: PresenceCount())
    anchor = tabular_anchor(tmp_path / "anchor", views.view(0))
    options = dict(kind="ridge", batch_size=3, input_policy=HISTORICAL_MASKED)
    for target in (views.view(0), views.view(1)):
        output = tmp_path / f"masked-{target.parent.name}"
        receipt = carry.carry_tabular(
            anchor, views.view(0), target, output, modality_ablation=variant, **options
        )
        assert set(receipt["predictions"]) == {"evaluation"}
        assert receipt["modality_ablation"] == ablation_identity(variant)
        # La identidad de la vista no cambia: la ablación se declara en su propio campo.
        assert receipt["manifest_sha256"] == sha256(target)
        ids, presence, targets, _ = reader(target, "evaluation")
        rows = pq.read_table(output / receipt["predictions"]["evaluation"]["path"]).to_pylist()
        assert [row["sample_id"] for row in rows] == ids and ids
        np.testing.assert_array_equal([row["target"] for row in rows], targets)
        removed = masked_bits(presence, variant).sum(axis=1)
        assert removed.any() and (removed == 0).any()
        np.testing.assert_array_equal(
            [row["prediction"] for row in rows], presence.sum(axis=1) - removed
        )
    normal = carry.carry_tabular(
        anchor, views.view(0), views.view(1), tmp_path / "plain", **options
    )
    assert "modality_ablation" not in normal and set(normal["predictions"]) == {
        "calibration",
        "evaluation",
    }


def predictions(output, receipt, partition="evaluation"):
    return pq.read_table(output / receipt["predictions"][partition]["path"])


def test_neural_ablation_changes_only_rows_with_a_masked_modality(views, tmp_path, cpu):  # noqa: F811
    neural_anchor(tmp_path / "anchor", views.view(0))
    options = dict(batch_size=2, input_policy=HISTORICAL_MASKED)
    anchor = tmp_path / "anchor"
    plain = carry.carry_reference(
        anchor, views.view(0), views.view(1), tmp_path / "plain", **options
    )
    assert "modality_ablation" not in plain
    original = predictions(tmp_path / "plain", plain)
    _, presence, _, _ = reader(views.view(1), "evaluation")
    for variant in VARIANTS:
        output = tmp_path / variant
        receipt = carry.carry_reference(
            anchor, views.view(0), views.view(1), output, modality_ablation=variant, **options
        )
        assert set(receipt["predictions"]) == {"evaluation"}
        assert receipt["modality_ablation"] == ablation_identity(variant)
        assert receipt["anchor"] == plain["anchor"] and receipt["fold"] == plain["fold"]
        masked = predictions(output, receipt)
        for name in ("sample_id", "market", "prediction_at", "target"):
            assert masked[name].equals(original[name])
        touched = masked_bits(presence, variant).any(axis=1)
        assert touched.any() and (~touched).any()
        for name in ("prediction", *QUANTILE_COLUMNS):
            before, after = original[name].to_numpy(), masked[name].to_numpy()
            np.testing.assert_array_equal(after[~touched], before[~touched])
        changed = original["prediction"].to_numpy() != masked["prediction"].to_numpy()
        assert changed[touched].any()
    # La propia ventana del ancla se predice solo con la ablación.
    own = carry.carry_reference(
        anchor,
        views.view(0),
        views.view(0),
        tmp_path / "own",
        modality_ablation="mask_news",
        **options,
    )
    assert own["months_since_anchor_information"] == 3
    with pytest.raises(ValueError, match="no es posterior"):
        carry.carry_reference(anchor, views.view(0), views.view(0), tmp_path / "again", **options)


@pytest.fixture(scope="module")
def titans(tmp_path_factory):
    """Ventanas Titans-MAC sobre el mismo corpus con y sin noticias y fundamentales reales.

    El optimizador inyectado solo registra gradientes, así que las dos ventanas conservan
    los parámetros iniciales de la semilla.
    """
    root = tmp_path_factory.mktemp("titans-ablation")
    hold = root / "training-hold.json"
    hold.write_text(json.dumps({"training_allowed": True}), encoding="utf-8")
    runs = {}
    with pytest.MonkeyPatch.context() as patch, unfused_attention():
        patch.setenv(HOLD_ENV, str(hold))
        path = recipe(root)
        for name, absent in (("original", ()), ("absent", ("news", "fundamentals"))):
            view, protocol = titans_views(root / name, absent=absent)
            report, _ = window(view, protocol, path, root / name / "run")
            runs[name] = SimpleNamespace(view=view, output=root / name / "run", report=report)
    return SimpleNamespace(root=root, hold=hold, **runs)


@pytest.fixture
def permitted(titans, monkeypatch):
    monkeypatch.setenv(HOLD_ENV, str(titans.hold))


def titans_carry(run, output, variant):
    with unfused_attention():
        return wf.carry_titans(
            run.output, run.view, run.view, output, device="cpu", modality_ablation=variant
        )


def same_rows(left, right):
    left, right = left.sort_by("sample_id"), right.sort_by("sample_id")
    for name in ("sample_id", "prediction_at", "target"):
        assert left[name].equals(right[name])
    return left, right


def test_titans_ablation_of_absent_modalities_reproduces_the_window(titans, tmp_path, permitted):
    run = titans.absent
    fitted = pq.read_table(run.output / run.report["predictions"]["evaluation"]["path"])
    receipt = titans_carry(run, tmp_path / "masked", "mask_news_and_fundamentals")
    assert set(receipt["predictions"]) == {"evaluation"} and set(receipt["phases"]) == {
        "evaluation"
    }
    assert receipt["modality_ablation"] == ablation_identity("mask_news_and_fundamentals")
    assert receipt["anchor"]["checkpoint_sha256"] == run.report["checkpoint"]["sha256"]
    masked = predictions(tmp_path / "masked", receipt)
    fitted, masked = same_rows(fitted, masked)
    for name in ("prediction", *QUANTILE_COLUMNS):
        np.testing.assert_array_equal(masked[name].to_numpy(), fitted[name].to_numpy())


def test_titans_ablation_equals_the_window_with_a_real_absence(titans, tmp_path, permitted):
    """La memoria rápida ve la ausencia desde el calentamiento, como en el corpus sin ellas."""
    absent = pq.read_table(
        titans.absent.output / titans.absent.report["predictions"]["evaluation"]["path"]
    )
    receipt = titans_carry(titans.original, tmp_path / "masked", "mask_news_and_fundamentals")
    masked = predictions(tmp_path / "masked", receipt)
    original = pq.read_table(
        titans.original.output / titans.original.report["predictions"]["evaluation"]["path"]
    )
    absent, masked = same_rows(absent, masked)
    original, _ = same_rows(original, masked)
    for name in ("prediction", *QUANTILE_COLUMNS):
        np.testing.assert_array_equal(masked[name].to_numpy(), absent[name].to_numpy())
    assert (masked["prediction"].to_numpy() != original["prediction"].to_numpy()).any()


def test_titans_warmup_and_evaluation_read_only_masked_inputs(
    titans, tmp_path, monkeypatch, permitted
):
    seen = []
    event = FinancialObservationSource._event

    def recorded(self, at, rows, reader=None):
        result = event(self, at, rows, reader)
        seen.extend((at, self.phase, batch) for batch in result.inputs)
        return result

    monkeypatch.setattr(FinancialObservationSource, "_event", recorded)
    titans_carry(titans.original, tmp_path / "masked", "mask_news")
    phase = seen[0][1]
    assert phase.partition == "evaluation" and phase.warmup_start < phase.decision_start
    warmup = [batch for at, _, batch in seen if at < phase.decision_start]
    measured = [batch for at, _, batch in seen if at >= phase.decision_start]
    assert warmup and measured
    news = MODALITIES.index("news")
    for batch in warmup + measured:
        assert not batch["presence"][:, news].any() and not batch["inputs"]["news"].any()
    # Sin ablación, el mismo calentamiento sí contiene noticias.
    plain = FinancialObservationSource(
        wf.CorpusDataset(titans.original.view, input_policy=HISTORICAL_MASKED),
        next((tmp_path / "masked" / "indices").glob("evaluation-*/manifest.json")),
    )
    batches = [batch for item in plain.events() for batch in item.inputs]
    assert any(batch["presence"][:, news].any() for batch in batches)


@pytest.fixture(scope="module")
def corrections(titans):
    """Corrección B6 proximal con la clave del codec sobre las dos ventanas Titans-MAC."""
    runs = {}
    with pytest.MonkeyPatch.context() as patch, unfused_attention():
        patch.setenv(HOLD_ENV, str(titans.hold))
        for name in ("original", "absent"):
            run = getattr(titans, name)
            output = titans.root / name / "b6"
            report = mc.run_correction_window(
                run.view,
                run.output,
                CORRECTION,
                components=B6,
                seed=42,
                output=output,
                search_case="eta25e-2",
                device="cpu",
            )
            assert report["status"] == "completed"
            runs[name] = SimpleNamespace(view=run.view, output=output, report=report)
    return SimpleNamespace(**runs)


def test_b6_ablation_equals_the_correction_with_a_real_absence(corrections, tmp_path, permitted):
    """El núcleo, el calentamiento y las claves del codec ven la ausencia, y A se escribe con
    los errores de ese recorrido. La ablación llega por el traslado común de la familia."""
    run = corrections.original
    with unfused_attention():
        receipt = mw.carry_mars_titan_arm(
            run.output,
            run.view,
            run.view,
            tmp_path / "masked",
            device="cpu",
            modality_ablation=BOTH,
        )
    # La etapa de ablación usa este mismo traslado para toda la familia.
    assert ablation_stage._mars_titan() is mw.carry_mars_titan_arm
    assert receipt["kind"] == mc.CARRY_KIND and set(receipt["predictions"]) == {"evaluation"}
    assert receipt["modality_ablation"] == ablation_identity(BOTH)
    assert receipt["anchor"]["components"] == B6
    masked = predictions(tmp_path / "masked", receipt)
    absent = predictions(corrections.absent.output, corrections.absent.report)
    original = predictions(run.output, run.report)
    absent, masked = same_rows(absent, masked)
    original, _ = same_rows(original, masked)
    for name in ("prediction", *QUANTILE_COLUMNS):
        np.testing.assert_array_equal(masked[name].to_numpy(), absent[name].to_numpy())
    assert (masked["prediction"].to_numpy() != original["prediction"].to_numpy()).any()
