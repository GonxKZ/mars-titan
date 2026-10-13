"""Sensibilidad de la comparación al preentrenamiento posterior de los codificadores.

MiniLM y ResNet18 se entrenaron con textos e imágenes reunidos hasta la publicación de sus
pesos. La edición congelada representa una noticia de 2005 con un modelo que pudo leer cómo
terminó aquella historia. El codificador de control (`data.pretraining_free_encoders`) no
tiene parámetros aprendidos. La sensibilidad vuelve a ajustar los modelos declarados sobre una
edición con ese control, con las mismas filas, ventanas, semillas y configuración, y compara
las dos comparaciones walk-forward publicadas.

Para cada modelo y mercado, Δ de una sesión es el MAE con el control menos el MAE con los
codificadores congelados, después de promediar las semillas sesión a sesión. Positivo indica
que los codificadores congelados ayudan. Los estadísticos son medias de Δ por tramos de 30
meses alrededor del corte, el primer mes completo después de publicar los pesos de MiniLM:

- `effect`: media de todas las sesiones evaluadas.
- `break`: tramo anterior al corte menos tramo posterior. Si la ventaja viniera de conocer el
  futuro, existiría antes del corte y no después.
- `placebo_break`: el mismo salto un tramo antes, entre dos tramos que el codificador pudo
  conocer. Una ventaja que solo decae con el tiempo produce los dos saltos por igual.

La anticipación se considera sospechosa si `break` y `break − placebo_break` quedan por encima
de cero con intervalos simultáneos. En otro caso no se sostiene. Con menos sesiones de las
declaradas en algún tramo la decisión queda sin tomar. Los intervalos usan el bootstrap
circular por bloques de días UTC declarado, igual al de la comparación, con una familia max-t
por mercado sobre los modelos y los dos estadísticos de la decisión.

No se ha ejecutado: necesita la edición de control y los ajustes de los modelos declarados,
que esperan al desbloqueo del aprendizaje.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.paired_comparisons import circular_block_counts, family_intervals
from mars_titan.evaluation.walk_forward_comparison import REPORT_KIND as COMPARISON_KIND

KIND = "pretraining_free_encoder_sensitivity"
REPORT_KIND = "pretraining_free_encoder_sensitivity_report"
STATUS = "declared_before_execution"
USE = "sensitivity_not_for_model_selection_or_primary_conclusion"
METRIC = "session_mae_control_encoders_minus_frozen_encoders_after_averaging_seeds"
STATISTICS = {
    "effect": "mean_of_all_evaluated_sessions",
    "break": "block_before_the_cutoff_minus_block_after_the_cutoff",
    "placebo_break": "block_two_before_the_cutoff_minus_block_before_the_cutoff",
}
DECISION = "anticipation_suspected_if_the_break_and_the_break_minus_the_placebo_break_exceed_zero"
MULTIPLICITY = "max_t_over_arms_and_both_decision_statistics_within_each_market"
DOES_NOT_MEASURE = [
    "which_of_the_two_frozen_encoders_carries_the_effect",
    "regime_changes_that_coincide_with_the_cutoff",
    "knowledge_of_the_world_that_does_not_change_the_error",
]
FIELDS = {
    "schema_version",
    "kind",
    "issue",
    "status",
    "declared_at",
    "use",
    "hypothesis",
    "frozen_encoders",
    "control_encoders",
    "knowledge_bound",
    "cutoff",
    "block_months",
    "arms",
    "scopes",
    "metric",
    "statistics",
    "decision",
    "min_sessions",
    "bootstrap",
    "multiplicity",
    "does_not_measure",
}
_BOOTSTRAP = {"block_length", "replicates", "seed", "confidence"}
_BLOCKS = ("placebo", "before", "after")
_DAY = 86_400_000_000
_MAX_BYTES = 256 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _iso(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def declaration(section):
    """Validar la declaración previa sin abrir ningún resultado."""
    _require(
        isinstance(section, dict) and set(section) == FIELDS,
        "La sensibilidad de los codificadores no tiene exactamente sus campos",
    )
    _require(
        section["schema_version"] == 1
        and section["kind"] == KIND
        and section["status"] == STATUS
        and section["use"] == USE
        and _iso(section["declared_at"])
        and isinstance(section["hypothesis"], str)
        and section["hypothesis"].strip()
        and section["does_not_measure"] == DOES_NOT_MEASURE,
        "La sensibilidad debe declararse antes de ejecutar, con su hipótesis y sus límites",
    )
    _require(
        section["metric"] == METRIC
        and section["statistics"] == STATISTICS
        and section["decision"] == DECISION
        and section["multiplicity"] == MULTIPLICITY,
        "La sensibilidad usa Δ por sesión, el salto en el corte y su placebo",
    )
    cutoff, months = section["cutoff"], section["block_months"]
    _require(
        _iso(cutoff)
        and cutoff.endswith("-01")
        and type(months) is int
        and 1 <= months <= 120
        and type(section["min_sessions"]) is int
        and section["min_sessions"] >= 1,
        "El corte debe empezar un mes y los tramos y mínimos deben ser enteros acotados",
    )
    bound = section["knowledge_bound"]
    _require(
        isinstance(bound, dict)
        and _iso(bound.get("text_weights_published"))
        and bound["text_weights_published"] < cutoff,
        "El corte debe quedar después de la publicación verificada de los pesos",
    )
    arms, scopes = section["arms"], section["scopes"]
    _require(
        isinstance(arms, list)
        and arms
        and len(set(arms)) == len(arms)
        and all(isinstance(arm, str) and arm for arm in arms)
        and isinstance(scopes, list)
        and scopes
        and set(scopes) <= {"US", "CN", "US+CN"}
        and len(set(scopes)) == len(scopes),
        "La sensibilidad necesita modelos y ámbitos declarados sin repetir",
    )
    options = section["bootstrap"]
    _require(
        isinstance(options, dict)
        and set(options) == _BOOTSTRAP
        and type(options["block_length"]) is int
        and options["block_length"] >= 1
        and type(options["replicates"]) is int
        and options["replicates"] >= 100
        and type(options["seed"]) is int
        and 0 < options["confidence"] < 1,
        "El bootstrap de la sensibilidad no es válido",
    )
    return section


def load_declaration(path):
    section, _ = read_manifest(Path(path))
    return declaration(section)


def _month(day, months):
    position = day.year * 12 + day.month - 1 + months
    return date(position // 12, position % 12 + 1, 1)


def block_edges(section):
    """Inicios del placebo, del tramo anterior, del corte y fin del posterior, en µs UTC."""
    cutoff = date.fromisoformat(section["cutoff"])
    months = section["block_months"]
    days = [_month(cutoff, k * months) for k in (-2, -1, 0, 1)]
    return [int(np.datetime64(day, "us").astype(np.int64)) for day in days]


def read_comparison(folder):
    """Informe y tabla por sesión de una comparación walk-forward publicada."""
    folder = Path(folder)
    report, digest = read_manifest(folder / "comparison.json", _MAX_BYTES)
    _require(
        report.get("kind") == COMPARISON_KIND
        and report.get("status") == "completed"
        and report.get("final_test_opened") is False,
        "La comparación no está completa o abrió el test final",
    )
    path = folder / "sessions.parquet"
    _require(
        path.is_file() and sha256(path) == report.get("artifacts", {}).get("sessions.parquet"),
        "La tabla por sesión no coincide con su informe",
    )
    table = pq.read_table(path)
    table = table.filter(pc.equal(table["quantiles"], "raw"))
    return report, digest, table


def _check_pair(frozen, control, section):
    _require(
        frozen["scope"] == control["scope"] and frozen["scope"] in section["scopes"],
        "Las dos comparaciones deben evaluar el mismo ámbito declarado",
    )
    _require(
        frozen["configuration"] == control["configuration"]
        and frozen["markets"] == control["markets"]
        and {k: v["evaluation"] for k, v in frozen["windows"].items()}
        == {k: v["evaluation"] for k, v in control["windows"].items()},
        "Las dos comparaciones deben usar la misma configuración, mercados y ventanas",
    )
    _require(frozen["edition"] != control["edition"], "Las dos comparaciones usan la misma edición")


def _rows(table, arm):
    """Filas de un modelo ordenadas por mercado, sesión, semilla y ventana."""
    columns = ("market", "prediction_at", "seed", "window")
    rows = table.filter(pc.equal(table["arm"], arm))
    _require(rows.num_rows > 0, f"El modelo {arm} no está en las dos comparaciones")
    order = pc.sort_indices(rows, sort_keys=[(name, "ascending") for name in columns])
    return rows.take(order)


def session_deltas(frozen, control, arm, market):
    """Instantes, días UTC y Δ por sesión de un modelo y mercado, con las semillas promediadas.

    Las dos tablas deben tener las mismas sesiones, semillas y ventanas, el mismo número de
    filas por sesión y los mismos signos de los objetivos.
    """
    tables = [_rows(table, arm) for table in (frozen, control)]
    keys = ("market", "prediction_at", "seed", "window")
    same = ("samples", "positive_targets", "negative_targets")
    _require(
        all(tables[0][k].equals(tables[1][k]) for k in keys + same),
        f"El modelo {arm} no evalúa las mismas filas en las dos ediciones",
    )
    mask = pc.equal(tables[0]["market"], market)
    times = pc.filter(tables[0]["prediction_at"], mask).cast("int64").to_numpy()
    seeds = pc.filter(tables[0]["seed"], mask).to_numpy()
    mae = [pc.filter(table["mae"], mask).to_numpy().astype(np.float64) for table in tables]
    _require(len(times) > 0, f"El modelo {arm} no tiene sesiones de {market}")
    unique, inverse = np.unique(times, return_inverse=True)
    count = len(np.unique(seeds))
    _require(
        np.all(np.bincount(inverse) == count),
        f"Alguna sesión de {arm} en {market} no tiene todas sus semillas",
    )
    # Media de las semillas en cada sesión, igual que las series de los contrastes.
    means = [np.bincount(inverse, weights=values) / count for values in mae]
    delta = means[1] - means[0]
    days = np.unique(np.floor_divide(unique, _DAY), return_inverse=True)[1]
    return unique, days, delta


def _cells(times, days, delta, edges):
    """Sumas y recuentos diarios de todas las sesiones y de los tres tramos."""
    masks = [np.ones(len(times), bool)]
    masks += [(times >= low) & (times < high) for low, high in zip(edges, edges[1:], strict=False)]
    periods = int(days.max()) + 1
    sums = np.stack(
        [np.bincount(days, weights=delta * m, minlength=periods) for m in masks], axis=1
    )
    sizes = np.stack(
        [np.bincount(days, weights=m.astype(np.float64), minlength=periods) for m in masks], axis=1
    )
    return sums, sizes


def _statistics(sums, sizes):
    """effect, break y placebo_break a partir de sumas y recuentos (última dimensión)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(sizes > 0, sums / np.where(sizes > 0, sizes, 1.0), np.nan)
    whole, placebo, before, after = (means[..., k] for k in range(4))
    return np.stack([whole, before - after, placebo - before], axis=-1)


def _float(value):
    return float(value) if np.isfinite(value) else None


def market_report(section, arms):
    """Estimaciones, intervalos y decisión de cada modelo de un mercado.

    `arms` asigna a cada modelo sus vectores (instantes, días, Δ). Todos los modelos de un
    mercado se remuestrean con los mismos bloques de días.
    """
    options = section["bootstrap"]
    edges = block_edges(section)
    sessions = {tuple(value[0]) for value in arms.values()}
    _require(len(sessions) == 1, "Los modelos de un mercado no evalúan las mismas sesiones")
    periods = int(next(iter(arms.values()))[1].max()) + 1
    cells = {arm: _cells(*value, edges) for arm, value in arms.items()}
    draws = None
    if options["block_length"] < periods:
        rng = np.random.default_rng(options["seed"])
        counts = circular_block_counts(rng, options["replicates"], periods, options["block_length"])
        counts = counts.astype(np.float64)
        draws = {arm: _statistics(counts @ s, counts @ z) for arm, (s, z) in cells.items()}
    result, family = {}, []
    for arm, (sums, sizes) in cells.items():
        estimate = _statistics(sums.sum(axis=0), sizes.sum(axis=0))
        totals = sizes.sum(axis=0).astype(int)
        record = dict(
            sessions=dict(zip(("all", *_BLOCKS), totals.tolist(), strict=True)),
            estimates={key: _float(estimate[i]) for i, key in enumerate(STATISTICS)},
            intervals={key: None for key in STATISTICS},
            decision=dict(status="undetermined", intervals=None, reason=None),
        )
        result[arm] = record
        if draws is not None:
            for i, key in enumerate(STATISTICS):
                column = draws[arm][:, i]
                if np.isfinite(column).all():
                    bounds = family_intervals(
                        estimate[i : i + 1], column[:, None], options["confidence"]
                    )
                    record["intervals"][key] = [
                        float(bounds["lower"][0]),
                        float(bounds["upper"][0]),
                    ]
        if draws is None:
            record["decision"]["reason"] = "Se necesitan más días que la longitud del bloque"
        elif min(totals[1:]) < section["min_sessions"]:
            record["decision"]["reason"] = "Algún tramo tiene menos sesiones de las declaradas"
        elif not np.isfinite(draws[arm][:, 1:]).all():
            record["decision"]["reason"] = "Alguna réplica se queda sin sesiones en un tramo"
        else:
            jump, placebo = estimate[1], estimate[1] - estimate[2]
            columns = (draws[arm][:, 1], draws[arm][:, 1] - draws[arm][:, 2])
            family.append((arm, (jump, placebo), columns))
    if family:
        estimate = np.array([value for _, values, _ in family for value in values])
        replicas = np.stack([column for *_, columns in family for column in columns], axis=1)
        joint = family_intervals(estimate, replicas, options["confidence"])
        for position, (arm, _, _) in enumerate(family):
            bounds = {
                label: [
                    float(joint["joint_lower"][2 * position + k]),
                    float(joint["joint_upper"][2 * position + k]),
                ]
                for k, label in enumerate(("break", "break_minus_placebo_break"))
            }
            above = [low > 0 for low, _ in bounds.values()]
            reason = None
            if not above[0]:
                reason = "El intervalo del salto en el corte no queda por encima de cero"
            elif not above[1]:
                reason = "El salto en el corte no se separa del salto placebo"
            result[arm]["decision"] = dict(
                status="anticipation_suspected" if all(above) else "not_supported",
                intervals=bounds,
                critical=joint["critical"],
                reason=reason,
            )
    return result


def evaluate(section, frozen_folder, control_folder):
    """Informe de la sensibilidad para el ámbito de las dos comparaciones."""
    section = declaration(section)
    frozen, frozen_sha, frozen_table = read_comparison(frozen_folder)
    control, control_sha, control_table = read_comparison(control_folder)
    _check_pair(frozen, control, section)
    markets = {}
    for market in frozen["markets"]:
        arms = {
            arm: session_deltas(frozen_table, control_table, arm, market) for arm in section["arms"]
        }
        markets[market] = market_report(section, arms)
    return dict(
        schema_version=1,
        kind=REPORT_KIND,
        final_test_opened=False,
        declaration=section,
        scope=frozen["scope"],
        comparisons=dict(
            frozen=dict(sha256=frozen_sha, edition=frozen["edition"]),
            control=dict(sha256=control_sha, edition=control["edition"]),
        ),
        markets=markets,
        code_sha256=sha256(Path(__file__)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True, help="Comparación congelada")
    parser.add_argument("--control", type=Path, required=True, help="Comparación de control")
    parser.add_argument("--output", type=Path, required=True, help="Informe nuevo")
    args = parser.parse_args(argv)
    safe_destination(args.output)
    for source in (args.frozen, args.control):
        outside_source(source, args.output)
    _require(not args.output.exists(), "El informe necesita un archivo nuevo")
    report = evaluate(load_declaration(args.declaration), args.frozen, args.control)
    atomic_json(args.output, report)
    print(json.dumps(dict(output=str(args.output), scope=report["scope"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
