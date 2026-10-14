"""Preprocesado ajustado solo con el pasado de cada corte (#442).

Normalizaciones, residualizadores, umbrales de régimen y estratos solo pueden usar datos
anteriores al tramo en el que se aplican. El informe de integridad de la publicación reúne
estas comprobaciones en la sección `preprocessing`.

**Código.** Una transformación que solo mira al pasado da el mismo valor en una decisión
aunque cambien los precios posteriores. `past_only` recalcula con los precios posteriores a
cada corte alterados y compara bit a bit los valores hasta el corte. `audit_code` lo aplica
a las dos implementaciones del residualizador de las etiquetas y a las rutas de régimen, con
precios deterministas construidos aquí que no alimentan ningún modelo. Las pruebas
comprueban que una versión ajustada con datos futuros (beta de toda la muestra, ventana que
mira sesiones posteriores) la incumple.

**Edición.** Las etiquetas residuales y el calendario de regímenes se calculan una vez, fuera
de la campaña. `check_editions` exige que la edición que leen las comparaciones publicadas
declare el código de sus etiquetas y que sea el mismo que pasa `audit_code`, y lo mismo para
el calendario de regímenes. Con la política `real_edition_only` (A v2) una edición o un
calendario sin código declarado hacen fallar la comprobación. Las campañas sin esa política,
como las reducidas de las pruebas con etiquetas escritas a mano, lo informan sin exigirlo.

**Ajustes.** `check_fits` lee las estadísticas que guardó cada ajuste confirmado:
- Ridge y XGBoost: medias, escalas y cortes de histograma recorren exactamente las filas de
  entrenamiento (`fitted_rows` igual a `samples.train`);
- MARS-TITAN con M3: las escalas de escritura salen del índice de entrenamiento de la ventana,
  con sus mismos límites de decisión, y esas etiquetas maduran antes de la primera decisión
  de validación. Los demás bancos no llevan escalas;
- las demás familias no estiman estadísticas con datos (`WITHOUT_FITTED`, con el motivo), y
  una familia sin inventariar hace fallar la comprobación.

`check_stage` hace lo mismo con la normalización de cada padre de la etapa de adaptadores,
que solo recorre las filas reales de entrenamiento y cuyas etiquetas maduran antes de
validación.

**Reglas declaradas.** `check_rules` exige que cada sección de la comparación esté en el
inventario `SECTIONS` con su regla causal, o sin regla ajustada con datos. Una sección nueva
hace fallar la comprobación hasta que se declare.

Solo lee declaraciones, recibos, informes y manifiestos. No ajusta nada ni abre la reserva
de 2024.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation import regime_calendar
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.memory.regimes import REGIME_RULE, RegimeRule
from mars_titan.training import campaign_plan as plan
from mars_titan.training.campaign_data_policy import REAL_EDITION_ONLY
from mars_titan.training.temporal_contract import temporal_contracts
from mars_titan.training.walk_forward_phases import micros

from . import fit_budget

KIND = "past_only_preprocessing"
CONFIRMED = ("completed",)
ROOT = Path(__file__).parents[1]
# Código de las etiquetas residuales que declara la edición, por campo de su configuración.
TARGET_CODE = {
    "target_implementation_sha256": "data/residual_arrays.py",
    "target_reference_sha256": "data/budget_targets.py",
    "temporal_reference_sha256": "data/temporal.py",
}
# Familias de ajuste de la campaña sin estadísticas estimadas con datos, con el motivo.
WITHOUT_FITTED = {
    plan.NEURAL: "Las entradas llegan codificadas por la edición y el modelo no estima escalas",
    plan.TITANS: "La memoria escribe con sus entradas, pero no hay escalas ni umbrales ajustados",
    plan.EPISODIC: "El códec normaliza cada fila y proyecta con una matriz fija",
    plan.CM: "Los núcleos y el factorial no estiman escalas ni umbrales con datos",
}
# Secciones de la comparación y regla que las hace causales, o None si no ajustan nada.
SECTIONS = {
    "schema_version": None,
    "kind": None,
    "name": None,
    "status": None,
    "input_policy": None,
    "partition": None,
    "scopes": None,
    "arms": None,
    "metrics": None,
    "comparison": None,
    "predictive_ability": None,
    "joint_design": None,
    "modality_ablation": None,
    # Calibración conformal ajustada en el tramo de calibración, anterior a la evaluación.
    "calibration": ("partition", {"calibration"}),
    # Presencia de cada modalidad en la muestra de la decisión, sin umbrales estimados.
    "modality_strata": ("presence_source", {"view_evaluation_labels_and_samples_presence_v1"}),
    # Cuartiles de las predicciones de la misma sesión.
    "long_short": ("kind", {"quartile_long_short_v1"}),
    # Calendario con ventanas de precios que terminan en cada decisión.
    "retention_interference": ("regimes", {"observable_volatility_trend_v1_market_calendar"}),
    # Umbrales fijos de precio y volumen sobre la edición sin ajustar hasta la última sesión
    # publicada al emitir (``tests/evaluation/test_liquidity_strata.py`` altera las posteriores).
    "liquidity_strata": (
        "source",
        {"unadjusted_reconstructed_verified_rows_up_to_the_last_published_session_v1"},
    ),
    # Calibración en línea y su reajuste periódico: una etiqueta solo interviene en una emisión
    # posterior a su `target_available_at` de la vista (`tests/calibration/test_online_conformal.py`
    # y `test_periodic_conformal.py` alteran las etiquetas futuras e inmaduras).
    "online_calibration": ("maturity_source", {"view_labels_target_available_at_v1"}),
}
# Cortes y alteración de los precios posteriores de la auditoría del código.
AUDIT_CUTS = (0.45, 0.7, 0.9)
AUDIT_SHOCK = (1.7, 0.6)


def _same(left, right):
    """Igualdad bit a bit de dos columnas, con los ausentes en las mismas filas."""
    left, right = np.asarray(left, dtype=object), np.asarray(right, dtype=object)
    missing = pd.isna(left)
    return (
        left.shape == right.shape
        and np.array_equal(missing, pd.isna(right))
        and all(a == b for a, b in zip(left[~missing], right[~missing], strict=True))
    )


def past_only(compute, data, perturb, cuts, groups):
    """Valores hasta cada corte que cambian al alterar solo los datos posteriores.

    `compute(data)` devuelve una tabla con una fila por decisión y el mismo orden con
    cualquier alteración. `perturb(data, cut)` devuelve una copia con los datos posteriores
    a `cut` alterados. `groups(table, cut)` da pares de columnas y máscara de las filas que no
    pueden depender de esos datos. Sin ninguna fila comparada la comprobación no pasa.
    """
    base = compute(data)
    changed, compared = [], 0
    for cut in cuts:
        table = compute(perturb(data, cut))
        if len(table) != len(base):
            changed.append(f"corte {cut}: {len(base)} filas antes y {len(table)} después")
            continue
        for columns, mask in groups(base, cut):
            compared += int(np.sum(mask)) * len(columns)
            changed += [
                f"corte {cut}: {column}"
                for column in columns
                if not _same(base.loc[mask, column], table.loc[mask, column])
            ]
    return dict(
        cuts=len(cuts), compared=compared, changed=changed, passed=compared > 0 and not changed
    )


def _shocked(values, after):
    """Copia con los valores posteriores a la posición `after` multiplicados por el choque."""
    values = np.array(values, dtype=np.float64)
    factors = np.resize(np.asarray(AUDIT_SHOCK), values.shape[-1] - after - 1)
    values[..., after + 1 :] *= factors
    return values


def residual_history():
    """Precios deterministas de un activo y su factor con la forma que lee el residualizador."""
    clock = MarketClock("US", "2021-01-01", "2023-12-31")
    rng = np.random.default_rng(442)
    factor = rng.normal(0.0, 0.01, len(clock.days))
    stock = 0.0002 + 1.3 * factor + rng.normal(0.0, 0.008, len(clock.days))
    common = dict(
        session=[day.isoformat() for day in clock.days], open=100.0, available_at=clock.decisions
    )
    market = pd.DataFrame({**common, "close": 100.0 * np.cumprod(1.0 + factor)})
    asset = pd.DataFrame({**common, "close": 50.0 * np.cumprod(1.0 + stock)})
    return clock, asset, market


def audit_residualizer(implementation, *, history=None):
    """Alfa, beta, historia y motivo hasta cada decisión, y objetivos ya maduros, fijos."""
    clock, asset, market = history or residual_history()
    cuts = [int(len(clock.days) * fraction) for fraction in AUDIT_CUTS]

    def compute(data):
        return implementation(data[0], data[1], clock, cutoff="2023-12-31")

    def perturb(data, cut):
        return tuple(frame.assign(close=_shocked(frame["close"].to_numpy(), cut)) for frame in data)

    def groups(table, cut):
        decision = pd.Timestamp(clock.decisions[cut])
        matured = pd.to_datetime(table["target_available_at"], utc=True)
        return [
            (("alpha", "beta", "history_pairs", "reason"), table["prediction_at"] <= decision),
            (("target", "target_available_at"), matured.notna() & (matured <= decision)),
        ]

    return past_only(compute, (asset, market), perturb, cuts, groups)


def regime_history():
    """Cierres deterministas de un mercado con los activos que pide la regla de régimen."""
    rng = np.random.default_rng(443)
    sessions = pd.bdate_range("2022-01-03", periods=200)
    days = [day.date().isoformat() for day in sessions]
    at = np.array([micros(day) + 21 * 3_600_000_000 for day in days], dtype=np.int64)
    drift = np.repeat(rng.normal(0.0, 0.004, (1, len(days))), 30, axis=0)
    closes = 100.0 * np.exp(np.cumsum(drift + rng.normal(0.0, 0.01, (30, len(days))), axis=1))
    return days, at, closes


def audit_regimes(routes, *, history=None):
    """Ruta, activos y rendimientos de cada sesión hasta el corte, fijos."""
    days, at, closes = history or regime_history()
    absent = np.zeros(len(days), dtype=bool)
    cuts = [int(len(days) * fraction) for fraction in AUDIT_CUTS]

    def compute(values):
        return pd.DataFrame(routes("US", days, at, values, absent, RegimeRule(REGIME_RULE)))

    def groups(table, cut):
        return [(("route", "assets", "returns"), table["at"] <= at[cut])]

    return past_only(compute, closes, _shocked, cuts, groups)


def audit_code():
    """Las implementaciones de las etiquetas y del régimen solo miran al pasado."""
    from mars_titan.data.budget_targets import residual_targets
    from mars_titan.data.residual_arrays import residual_targets_array

    result = dict(
        residual_reference=audit_residualizer(residual_targets),
        residual_arrays=audit_residualizer(residual_targets_array),
        regimes=audit_regimes(regime_calendar.market_routes),
    )
    return dict(result, passed=all(entry["passed"] for entry in result.values()))


def _code(names):
    return {field: sha256(ROOT / name) for field, name in names.items()}


def _parents(comparisons, policy):
    """Manifiestos padre de las vistas que leen las comparaciones walk-forward publicadas."""
    parents = {}
    for entry in comparisons:
        if not entry["label"].startswith("walk-forward/"):
            continue
        config = walk.resolve_config(entry["config"])
        sources = walk.load_sources(entry["sources"], config, entry["scope"])
        for path in sources["view_paths"].values():
            manifest, _ = read_manifest(path, 64 * 1024**2)
            for contract in temporal_contracts(manifest, input_policy=policy).values():
                parents[contract["parent_manifest"]] = contract["parent_sha256"]
    return parents


def check_editions(parents, calendar, *, strict):
    """Código de las etiquetas de cada edición y del calendario de regímenes publicado.

    `parents` asocia cada manifiesto padre de las vistas con su huella y `calendar` es la
    ruta del calendario o None. Con `strict`, la falta de código declarado también falla.
    """
    expected, failures, undeclared = _code(TARGET_CODE), [], []
    for path, digest in sorted(parents.items()):
        if not Path(path).is_file() or sha256(Path(path)) != digest:
            failures.append(f"{path}: la edición no es la que fijan sus vistas")
            continue
        configuration = read_manifest(Path(path), 8 * 1024**2)[0].get("configuration") or {}
        declared = {field: configuration.get(field) for field in TARGET_CODE}
        if not any(declared.values()):
            undeclared.append(path)
        elif declared != expected:
            failures.append(f"{path}: las etiquetas salen de otro código que el auditado")
    code = None
    if calendar is not None:
        document = read_manifest(Path(calendar), 64 * 1024**2)[0]
        regime_calendar.check(document)
        code = document.get("code_sha256")
        if code is None:
            undeclared.append(str(calendar))
        elif code != regime_calendar.code_sha256():
            failures.append(f"{calendar}: el calendario sale de otro código que el auditado")
    if strict:
        failures += [f"{path}: no declara el código con que se calculó" for path in undeclared]
        if not parents:
            failures.append("ninguna comparación publicada declara su edición")
    return dict(
        editions=sorted(parents),
        calendar=None if calendar is None else str(calendar),
        undeclared=undeclared,
        failures=failures,
        passed=not failures,
    )


def _train_rows(report):
    """Ridge y XGBoost: sus estadísticas recorren exactamente las filas de entrenamiento."""
    train = (report.get("samples") or {}).get("train")
    if type(train) is not int or train < 1:
        return ["el informe no declara sus filas de entrenamiento"]
    if report.get("fitted_rows") != train:
        return [f"las estadísticas recorren {report.get('fitted_rows')} filas de {train}"]
    return []


def _write_scalers(identity):
    """M3: escalas del índice de entrenamiento, con etiquetas maduras antes de validación."""
    scalers = (identity.get("retention") or {}).get("scalers")
    if identity.get("admission") != "m3":
        return [] if scalers is None else ["un banco sin M3 lleva escalas de escritura"]
    if not isinstance(scalers, dict):
        return ["M3 sin escalas de escritura"]
    sources = identity.get("sources") or {}
    train, validation = (sources.get(name) or {} for name in ("train", "validation"))
    phase, later = train.get("phase") or {}, validation.get("phase") or {}
    failures = []
    if (
        phase.get("partition") != "train"
        or scalers.get("source_sha256") != train.get("index_sha256")
        or scalers.get("dataset_sha256") != identity.get("dataset_sha256")
    ):
        failures.append("las escalas M3 no salen del índice de entrenamiento")
    if (scalers.get("decision_start"), scalers.get("decision_end")) != (
        phase.get("decision_start"),
        phase.get("decision_end"),
    ):
        failures.append("las escalas M3 cubren otras decisiones que el entrenamiento")
    closes, starts = phase.get("close_at"), later.get("decision_start")
    if type(closes) is not int or type(starts) is not int or closes > starts:
        failures.append("las etiquetas de entrenamiento maduran después de empezar validación")
    return failures


def _fit(job, output, receipt):
    family = job["family"]
    if family == plan.TABULAR:
        report, _ = fit_budget.fit_documents(output, receipt)
        return "train_rows", _train_rows(report)
    if family == plan.MARS:
        _, run = fit_budget.fit_documents(output, receipt)
        return "write_scalers", _write_scalers(run.get("identity") or {})
    if family in WITHOUT_FITTED:
        return "without_fitted", []
    return None, [f"la familia {family} no está en el inventario del preprocesado"]


def check_fits(jobs, output):
    """Estadísticas ajustadas con datos de cada ajuste confirmado de la campaña."""
    entries = []
    for job in jobs:
        if job["kind"] not in (plan.FIT, plan.ONLINE):
            continue
        receipt = fit_budget.read_receipt(output, job["id"])
        if receipt is None or receipt.get("status") not in CONFIRMED:
            # Los trabajos sin confirmar ya los cuenta `declared_runs`, y la meseta de un
            # ajuste conjunto tiene su propio estado y no guarda estadísticas nuevas.
            continue
        try:
            rule, failures = _fit(job, output, receipt)
        except ValueError as error:
            rule, failures = None, [str(error)]
        entries.append(dict(id=job["id"], family=job["family"], rule=rule, failures=failures))
    failing = [entry["id"] for entry in entries if entry["failures"]]
    return dict(fits=len(entries), failing=failing, jobs=entries, passed=not failing)


def _normalization(statistics, validation_start):
    """Normalización de un padre: filas reales de entrenamiento maduras antes de validación."""
    window = statistics.get("window") or {}
    samples, rows = statistics.get("samples"), window.get("train_rows")
    failures = []
    if statistics.get("fit_partition") != "train":
        failures.append("la normalización no se ajusta en el tramo de entrenamiento")
    if type(samples) is not int or samples < 1 or samples != rows:
        failures.append(f"la normalización recorre {samples} filas de {rows} de entrenamiento")
    bounds = window.get("train_bounds") or {}
    if not bounds or any(
        not isinstance(value, list) or len(value) != 3 or value[2] > validation_start
        for value in bounds.values()
    ):
        failures.append("las etiquetas de entrenamiento maduran después de empezar validación")
    return failures


def check_stage(campaign, planned, adapter_output):
    """Normalización de cada padre de la etapa de adaptadores con ajustes confirmados.

    Los ajustes cronológicos de la etapa reutilizan las escalas congeladas de su padre, que
    ya comprueba `check_fits`, y no estiman otras.
    """
    from mars_titan.posttraining import chronological_matrix

    windows = campaign["comparison_config"]["resolved_scopes"]
    entries, seen = [], set()
    for job in planned:
        key = tuple(job.get(name) for name in ("scope", "window", "base_arm", "seed"))
        if (
            job.get("kind") != plan.FIT
            or job.get("family") in chronological_matrix.CAMPAIGN_DESIGNS
            or key in seen
        ):
            continue
        receipt = fit_budget.read_receipt(adapter_output, job["id"])
        if receipt is None or receipt.get("status") not in CONFIRMED:
            continue
        seen.add(key)
        scope, window, base_arm, seed = key
        path = Path(adapter_output) / "windows-data" / scope / window / "parents"
        path = path / base_arm / f"seed-{seed}" / "normalization.json"
        if not path.is_file():
            failures = ["el padre no guarda su normalización"]
        else:
            start = micros(windows[scope]["windows"][window]["validation"][0])
            failures = _normalization(read_manifest(path, 4 * 1024**2)[0], start)
        entries.append(dict(parent="/".join(str(part) for part in key), failures=failures))
    failing = [entry["parent"] for entry in entries if entry["failures"]]
    return dict(parents=len(entries), failing=failing, entries=entries, passed=not failing)


def check_rules(path):
    """Cada sección de la comparación declarada está en el inventario con su regla causal."""
    document = read_manifest(Path(path), 1024**2)[0]
    failures, rules = [], {}
    for key, value in document.items():
        if key not in SECTIONS:
            failures.append(f"la sección {key} no está en el inventario del preprocesado")
            continue
        if SECTIONS[key] is None:
            continue
        field, allowed = SECTIONS[key]
        rules[key] = value.get(field) if isinstance(value, dict) else None
        if rules[key] not in allowed:
            failures.append(f"{key}: la regla {rules[key]} no está inventariada como causal")
    return dict(rules=rules, failures=failures, passed=not failures)


def check(
    campaign, jobs, output, comparisons, *, posttraining=None, adapter_output=None, regimes=None
):
    """Sección `preprocessing` del informe de integridad de la publicación."""
    strict = campaign.get("data_policy") == REAL_EDITION_ONLY
    sections = dict(
        code=audit_code(),
        editions=check_editions(
            _parents(comparisons, campaign["input_policy"]), regimes, strict=strict
        ),
        fits=check_fits(jobs, output),
        rules=check_rules(campaign["comparison_path"]),
    )
    if posttraining is not None:
        from mars_titan.posttraining import campaign_stage

        planned = campaign_stage.plan_stage(posttraining["stage"])
        sections["stage"] = check_stage(campaign, planned, adapter_output)
    return dict(
        kind=KIND,
        strict=strict,
        **sections,
        passed=all(entry["passed"] for entry in sections.values()),
    )
