"""Ablación de modalidades en la comparación walk-forward, con predicciones fijadas a mano.

Las predicciones originales y enmascaradas y los patrones de presencia se fijan en cada
prueba. No proceden de ningún modelo ajustado ni de la edición real, y no se ejecuta
ningún paso de optimizador.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.evaluation import modality_ablation as ablation
from mars_titan.evaluation import modality_strata as strata
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from tests.evaluation.test_comparison_sources import save
from tests.evaluation.test_modality_strata import by_asset, provider
from tests.evaluation.test_walk_forward_comparison import CONFIG, Study, rows

DECLARED = json.loads(CONFIG.read_text())
VARIANTS = list(ablation.VARIANTS)
# Desplazamiento de la predicción enmascarada en las filas afectadas. Titans no cambia.
SHIFTS = {
    "mask_news": dict(ridge=0.5, gru=0.25, titans=0.0),
    "mask_fundamentals": dict(ridge=-0.25, gru=0.5, titans=0.0),
    "mask_news_and_fundamentals": dict(ridge=0.75, gru=-0.5, titans=0.0),
}
VOLATILE = {"created_at_utc", "resources", "configuration", "analysis_source_sha256"}


def affected(variant, market, asset, moment):
    news, fundamentals = by_asset(market, asset, moment)
    return {
        "mask_news": news,
        "mask_fundamentals": fundamentals,
        "mask_news_and_fundamentals": news or fundamentals,
    }[variant]


def declare(study, version=3, **changes):
    """Versión 2 con estratos o versión 3 con estratos y ablación, con umbrales del fixture."""
    study.config["schema_version"] = version
    study.config["modality_strata"] = dict(
        copy.deepcopy(DECLARED["modality_strata"]), min_rows=6, min_sessions=3
    )
    study.config.pop("modality_ablation", None)
    if version == 3:
        study.config["modality_ablation"] = dict(
            copy.deepcopy(DECLARED["modality_ablation"]), min_rows=6, min_sessions=3, **changes
        )
    study.publish()
    return study


def masked_table(study, variant, arm, seed, fold, change=None):
    """Evaluación original con las filas afectadas desplazadas, columnas y orden iguales."""
    table = study.table(arm, seed, fold, "evaluation")
    keys, _ = rows(study.markets, int(fold["evaluation"][0][5:7]), fold["id"])
    mask = np.array([affected(variant, *key) for key in keys])
    shift = np.where(mask, SHIFTS[variant][arm], 0.0)
    values = table.to_pydict()
    values["prediction"] = list(np.asarray(values["prediction"]) + shift)
    for name in QUANTILE_COLUMNS:
        if name in values:
            values[name] = list(np.asarray(values[name]) + shift)
    if change is not None:
        change(variant, arm, seed, fold["id"], values)
    return pa.table(values, schema=table.schema)


def write_ablation(study, *, change=None, edit=None):
    """Escribir las predicciones enmascaradas y su manifiesto junto al estudio."""
    folder = study.root / "ablation"
    config = walk.load_config(study.config_path)
    variants = {}
    for variant in VARIANTS:
        for arm, declared in study.config["arms"].items():
            for seed in declared["seeds"]:
                for fold in study.folds:
                    relative = f"{variant}/{arm}/{seed}/{fold['id']}-evaluation.parquet"
                    path = folder / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(masked_table(study, variant, arm, seed, fold, change), path)
                    seeds = variants.setdefault(variant, {}).setdefault(arm, {})
                    seeds.setdefault(str(seed), {})[fold["id"]] = dict(
                        evaluation=dict(path=relative, sha256=sha256(path))
                    )
    manifest = dict(
        schema_version=1,
        kind=ablation.SOURCES_KIND,
        scope=study.scope,
        input_policy=study.policy,
        comparison_sha256=config["sha256"],
        windows={key: value["view"]["sha256"] for key, value in study.sources["windows"].items()},
        variants=variants,
    )
    if edit is not None:
        edit(manifest)
    save(folder / "sources.json", manifest)
    return folder / "sources.json"


def evaluate(study, monkeypatch, sources=None):
    monkeypatch.setattr(strata, "view_presence", provider(study, by_asset))
    return walk.evaluate_walk_forward(
        study.config_path, study.sources_path, study.scope, ablation_sources=sources
    )


def primary(report):
    return {key: value for key, value in report.items() if key not in VOLATILE}


@pytest.fixture(scope="module")
def joint(tmp_path_factory):
    """El mismo estudio con estratos (versión 2), con ablación pendiente y con ablación."""
    calls = []
    original = walk.cqr.fit_conformal_quantiles

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(walk.cqr, "fit_conformal_quantiles", counted)
        study = declare(Study(tmp_path_factory.mktemp("joint-ablation")), version=2)
        plain = evaluate(study, patch)
        fits = [len(calls)]
        declare(study)
        pending = evaluate(study, patch)
        fits.append(len(calls) - sum(fits))
        sources = write_ablation(study)
        report, sessions = evaluate(study, patch, sources)
        fits.append(len(calls) - sum(fits))
    return SimpleNamespace(
        study=study,
        plain=plain,
        pending=pending,
        report=report,
        sessions=sessions,
        sources=sources,
        fits=fits,
    )


def by_hand(study, variant, arm, seed, *, market=None):
    """MAE por sesión original y enmascarado en las filas afectadas, sin paneles."""
    sessions = {}
    for fold in study.folds:
        keys, target = rows(study.markets, int(fold["evaluation"][0][5:7]), fold["id"])
        before = study.table(arm, seed, fold, "evaluation")["prediction"].to_numpy()
        after = masked_table(study, variant, arm, seed, fold)["prediction"].to_numpy()
        for (m, a, t), y, p, q in zip(keys, target, before, after, strict=True):
            if not affected(variant, m, a, t) or (market and m != market):
                continue
            sessions.setdefault((m, t), []).append((abs(p - y), abs(q - y)))
    means = np.array([np.mean(errors, axis=0) for errors in sessions.values()])
    return means.mean(axis=0), len(sessions), sum(map(len, sessions.values()))


def test_declared_ablation_is_a_secondary_analysis_fixed_before_results():
    config = walk.load_config(CONFIG)
    section = config["modality_ablation"]
    assert config["schema_version"] == 3
    assert section == ablation.declaration(copy.deepcopy(section))
    assert section["status"] == "secondary_descriptive" and section["declared_at"] == "2026-10-09"
    assert "not_for_model_selection" in section["use"]
    assert section["variants"] == dict(
        mask_news=["news"],
        mask_fundamentals=["fundamentals"],
        mask_news_and_fundamentals=["news", "fundamentals"],
    )
    assert section["missing_reason"] == "modality_ablation" and section["recalibrate"] is False
    assert section["partition"] == "evaluation"
    assert section["metric"] == "session_mae_masked_minus_original"
    assert section["memory"] == "warmup_and_measured_partition_read_the_same_masked_inputs"
    assert section["does_not_measure"] == [
        "economic_causal_effect_of_the_modality",
        "usefulness_of_the_modality_for_the_forecast",
        "performance_of_a_model_trained_without_the_modality",
    ]
    assert (section["min_rows"], section["min_sessions"]) == (1000, 50)


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda s: s.pop("min_sessions"), "exactamente sus campos"),
        (lambda s: s.update(extra=1), "exactamente sus campos"),
        (lambda s: s.update(status="primary"), "secundario"),
        (lambda s: s.update(use="model_selection"), "secundario"),
        (lambda s: s.update(declared_at="2026-13-09"), "secundario"),
        (lambda s: s.update(declared_at="9 de octubre"), "secundario"),
        (lambda s: s.update(multiplicity="none"), "secundario"),
        (lambda s: s.update(does_not_measure=[]), "secundario"),
        (lambda s: s["variants"].update(mask_macro=["macro"]), "ausencia real"),
        (lambda s: s["variants"].update(mask_news=["news", "charts"]), "ausencia real"),
        (lambda s: s.update(masking="zero_vectors_only"), "ausencia real"),
        (lambda s: s.update(missing_reason="source_missing"), "ausencia real"),
        (lambda s: s.update(state="refit_without_the_modality"), "ausencia real"),
        (lambda s: s.update(memory="measured_partition_only"), "ausencia real"),
        (lambda s: s.update(partition="calibration"), "métrica"),
        (lambda s: s.update(rows="all_evaluation_rows"), "métrica"),
        (lambda s: s.update(metric="session_mae_ratio"), "métrica"),
        (lambda s: s.update(recalibrate=True), "volver a calibrar"),
        (lambda s: s.update(min_rows=0), "umbral"),
        (lambda s: s.update(min_sessions=True), "umbral"),
        (lambda s: s.update(min_rows=1.5), "umbral"),
    ],
)
def test_declaration_rejects_anything_but_the_declared_secondary_analysis(edit, message):
    section = copy.deepcopy(DECLARED["modality_ablation"])
    edit(section)
    with pytest.raises(ValueError, match=message):
        ablation.declaration(section)


def test_configuration_version_three_requires_its_ablation_section(tmp_path):
    document = copy.deepcopy(DECLARED)
    for scope in document["scopes"].values():
        scope["protocols"] = {
            market: str((CONFIG.parent / name).resolve())
            for market, name in scope["protocols"].items()
        }
    document.pop("modality_ablation")
    save(tmp_path / "missing.json", document)
    with pytest.raises(ValueError):
        walk.load_config(tmp_path / "missing.json")
    document["schema_version"] = 2
    save(tmp_path / "v2.json", document)
    assert "modality_ablation" not in walk.load_config(tmp_path / "v2.json")


def test_without_masked_sources_the_report_keeps_every_primary_output(joint, monkeypatch):
    plain, pending = joint.plain, joint.pending
    assert "modality_ablation" not in plain[0]
    assert primary(pending[0]) == primary(plain[0]) | dict(
        modality_ablation=pending[0]["modality_ablation"]
    )
    assert pending[1].equals(plain[1])
    section = pending[0]["modality_ablation"]
    assert section["status"] == "not_computed" and "Faltan" in section["reason"]
    assert section["declaration"] == joint.study.config["modality_ablation"]
    # Una configuración sin la ablación no admite predicciones enmascaradas.
    declare(joint.study, version=2)
    try:
        with pytest.raises(ValueError, match="no declara la ablación"):
            evaluate(joint.study, monkeypatch, joint.sources)
    finally:
        declare(joint.study)


def test_masked_sources_do_not_change_the_primary_report_nor_refit_any_calibrator(joint):
    pending, report = joint.pending[0], joint.report
    assert {key: value for key, value in primary(report).items() if key != "modality_ablation"} == {
        key: value for key, value in primary(pending).items() if key != "modality_ablation"
    }
    assert joint.sessions.equals(joint.pending[1])
    # Dos brazos de cuantiles con dos semillas en dos ventanas: ocho ajustes en cada informe.
    assert joint.fits == [8, 8, 8]
    section = report["modality_ablation"]
    assert section["status"] == "computed" and section["recalibrated"] is False
    assert section["sources_sha256"] == sha256(joint.sources)
    assert section["declaration"] == joint.study.config["modality_ablation"]
    names = ["US+CN", "US", "CN"]
    assert section["multiplicity"]["cells"] == len(VARIANTS) * len(names)
    assert section["multiplicity"]["confidence"] == strata.adjusted_confidence(0.95, 9)
    for name in ("evaluation/modality_ablation.py", "data/modality_ablation.py"):
        assert name in report["analysis_source_sha256"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_population_counts_only_rows_with_a_masked_modality(joint, variant):
    population = joint.report["modality_ablation"]["variants"][variant]["population"]
    total = 0
    for fold in joint.study.folds:
        keys, _ = rows(joint.study.markets, int(fold["evaluation"][0][5:7]), fold["id"])
        expected = sum(affected(variant, *key) for key in keys)
        window = population["windows"][fold["id"]]
        assert (window["rows"], window["unaffected_rows"]) == (expected, len(keys) - expected)
        for market in ("US", "CN"):
            cell = window["markets"][market]
            count = sum(affected(variant, *key) for key in keys if key[0] == market)
            assert cell["rows"] == count
        total += expected
    assert population["overall"]["US+CN"]["rows"] == total
    # Fundamentales en China: una fila por ventana, por debajo de los umbrales.
    china = population["overall"]["CN"]
    if variant == "mask_fundamentals":
        assert (china["rows"], china["sessions"], china["estimable"]) == (2, 2, False)
    else:
        assert china["estimable"] is True


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("arm,seed", [("ridge", 42), ("gru", 42), ("gru", 43), ("titans", 43)])
def test_difference_is_the_masked_minus_original_session_mae_on_affected_rows(
    joint, variant, arm, seed
):
    entry = joint.report["modality_ablation"]["variants"][variant]["arms"][arm][str(seed)]
    assert entry["unaffected_changed"] == {fold["id"]: 0 for fold in joint.study.folds}
    for market in (None, "US"):
        summary = entry["overall"][market or "US+CN"]
        (original, masked), _, _ = by_hand(joint.study, variant, arm, seed, market=market)
        assert summary["original_session_mae"] == pytest.approx(original, abs=1e-12)
        assert summary["masked_session_mae"] == pytest.approx(masked, abs=1e-12)
        assert summary["difference"] == pytest.approx(masked - original, abs=1e-12)
        if arm == "titans":
            assert summary["difference"] == 0.0
        if arm == "ridge":
            assert summary["calibrated_intervals"] is None
            assert summary["calibrated_reason"] == "El brazo no emite cuantiles"
        else:
            intervals = summary["calibrated_intervals"]
            assert [row["nominal"] for row in intervals["original"]] == [0.8, 0.95]
            if arm == "titans":
                # Sin cambio de predicción, el mismo calibrador da la misma cobertura.
                assert intervals["masked"] == intervals["original"]


def test_contrasts_estimate_each_arm_difference_within_the_variant_family(joint):
    for variant in VARIANTS:
        contrasts = joint.report["modality_ablation"]["variants"][variant]["contrasts"]
        family = contrasts["US+CN"]
        assert family["kind"] == "paired_session_contrasts" and family["metric"] == "mae"
        assert family["confidence"] == strata.adjusted_confidence(0.95, 9)
        estimates = {row["name"]: row for row in family["contrasts"]}
        assert set(estimates) == {"ridge", "gru", "titans"}
        assert estimates["titans"]["estimate"] == 0.0
        # Las semillas se promedian por sesión antes del contraste.
        differences = [by_hand(joint.study, variant, "gru", seed)[0] for seed in (42, 43)]
        expected = np.mean([masked - original for original, masked in differences])
        assert estimates["gru"]["estimate"] == pytest.approx(expected, abs=1e-12)
        if variant == "mask_fundamentals":
            assert contrasts["CN"] == dict(
                reason="2 filas y 2 sesiones, por debajo del mínimo declarado de 6 filas y 3 "
                "sesiones"
            )
        else:
            assert contrasts["CN"]["kind"] == "paired_session_contrasts"


def test_rows_without_the_modality_that_change_are_counted(joint, tmp_path, monkeypatch):
    study = Study(tmp_path / "changed")
    declare(study)
    first = study.folds[0]["id"]

    def change(variant, arm, seed, window, values):
        # Un brazo con memoria cambia una fila sin la modalidad: el activo G no tiene ninguna.
        if (arm, seed, window) == ("titans", 43, first):
            row = values["asset_id"].index("US/G")
            values["prediction"][row] += 1.0

    report, _ = evaluate(study, monkeypatch, write_ablation(study, change=change))
    for variant in VARIANTS:
        arms = report["modality_ablation"]["variants"][variant]["arms"]
        assert arms["titans"]["43"]["unaffected_changed"] == {first: 1, study.folds[1]["id"]: 0}
        assert set(arms["titans"]["42"]["unaffected_changed"].values()) == {0}
        # La fila cambiada no está en la métrica: la diferencia sigue siendo cero.
        assert arms["titans"]["43"]["overall"]["US+CN"]["difference"] == 0.0


def _drop_variant(manifest):
    manifest["variants"].pop("mask_news")


def _drop_arm(manifest):
    manifest["variants"]["mask_news"].pop("ridge")


def _drop_seed(manifest):
    manifest["variants"]["mask_news"]["gru"].pop("43")


def _drop_window(manifest):
    seeds = manifest["variants"]["mask_news"]["gru"]["42"]
    seeds.pop(next(iter(seeds)))


def _add_calibration(manifest):
    entry = next(iter(manifest["variants"]["mask_news"]["gru"]["42"].values()))
    entry["calibration"] = entry["evaluation"]


def _other_view(manifest):
    window = next(iter(manifest["windows"]))
    manifest["windows"][window] = "0" * 64


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda m: m.update(kind="walk_forward_prediction_sources"), "no cumplen su contrato"),
        (lambda m: m.update(scope="US"), "no cumplen su contrato"),
        (lambda m: m.update(extra=1), "no cumplen su contrato"),
        (lambda m: m.update(comparison_sha256="0" * 64), "otra política, comparación o vista"),
        (lambda m: m.update(input_policy="strict_four_modalities_v1"), "otra política"),
        (_other_view, "otra política, comparación o vista"),
        (_drop_variant, "exactamente las variantes"),
        (_drop_arm, "exactamente los brazos"),
        (_drop_seed, "exactamente las semillas"),
        (_drop_window, "exactamente las ventanas"),
        (_add_calibration, "solo declara su evaluación"),
    ],
)
def test_masked_sources_must_cover_exactly_the_declared_arms_seeds_and_windows(
    tmp_path, monkeypatch, edit, message
):
    study = declare(Study(tmp_path / "study"))
    sources = write_ablation(study, edit=edit)
    with pytest.raises(ValueError, match=message):
        evaluate(study, monkeypatch, sources)


@pytest.mark.parametrize(
    "change,message",
    [
        (
            lambda values: values["target"].__setitem__(0, values["target"][0] + 1),
            "no evalúa las mismas filas",
        ),
        (
            lambda values: values["prediction_at"].__setitem__(
                0, values["prediction_at"][0].replace(year=2022)
            ),
            "filas fuera del tramo de evaluation",
        ),
    ],
)
def test_masked_predictions_must_evaluate_the_same_rows_and_segment(
    tmp_path, monkeypatch, change, message
):
    study = declare(Study(tmp_path / "study"))

    def edited(variant, arm, seed, window, values):
        if (variant, arm, seed) == ("mask_fundamentals", "gru", 43):
            change(values)

    sources = write_ablation(study, change=edited)
    with pytest.raises(ValueError, match=message):
        evaluate(study, monkeypatch, sources)


def test_score_pair_rejects_another_cohort_and_counts_empty_cells():
    study = SimpleNamespace(cohort_sha256="a", rows=3, prediction=np.zeros(3))
    other = SimpleNamespace(cohort_sha256="b", rows=3, prediction=np.zeros(3))
    with pytest.raises(ValueError, match="mismas filas"):
        ablation.score_pair(study, other, np.zeros(3, bool), rank_ic_min_assets=3)
    with pytest.raises(ValueError, match="mismas filas"):
        ablation.score_pair(study, study, np.zeros(2, bool), rank_ic_min_assets=3)
    changed = SimpleNamespace(cohort_sha256="a", rows=3, prediction=np.array([0.0, 1.0, 0.0]))
    entry = ablation.score_pair(study, changed, np.zeros(3, bool), rank_ic_min_assets=3)
    assert entry == dict(
        rows=0,
        unaffected_rows=3,
        unaffected_changed=1,
        original=None,
        masked=None,
        calibrated_original=None,
        calibrated_masked=None,
    )
    bits = np.array([[1, 1, 1, 0, 1], [1, 0, 1, 1, 1], [1, 0, 1, 0, 1]], bool)
    assert ablation.affected_rows(bits, "mask_news").tolist() == [True, False, False]
    assert ablation.affected_rows(bits, "mask_fundamentals").tolist() == [False, True, False]
    both = ablation.affected_rows(bits, "mask_news_and_fundamentals")
    assert both.tolist() == [True, True, False]
