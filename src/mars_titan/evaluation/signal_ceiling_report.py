"""Informe del techo de señal (#747): los cuatro orígenes como una serie de 2019 a 2022.

Une en orden temporal las series por sesión de la evaluación de cada origen, como el análisis
de la criba, y publica para cada objetivo y modelo:

- El IC medio por sesión y el IC de rangos, con intervalos del bootstrap circular por bloques de
  días UTC de la criba, marginales y simultáneos dentro de su familia, y los valores p del paso
  descendente de Romano y Wolf y de Holm (`evaluate_family`).
- El MAE calibrado menos el del control cero, que predice un objetivo nulo, con la misma
  inferencia.
- La caída del IC al permutar cada familia de características dentro de cada sesión.
- El techo observado de cada objetivo: el modelo con mayor IC medio de validación, la media de
  los cuatro orígenes con el mismo peso, elegido sin mirar la evaluación. Un empate se queda con
  Ridge, el modelo más sencillo.

Las longitudes de bloque de sensibilidad se publican sin cambiar la lectura principal. El
módulo no ajusta modelos ni lee datos: recibe los informes ya escritos de cada origen.
"""

import numpy as np

from mars_titan.evaluation.variant_screening_analysis import evaluate_family

KIND = "signal_ceiling_report_v1"
MODELS = ("ridge", "lightgbm")
TARGETS = ("residual", "raw_open_close")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _joined(parts):
    """Series (valores, definidas) de los orígenes unidas en su orden."""
    return (
        np.concatenate([np.asarray(part["values"], dtype=np.float64) for part in parts]),
        np.concatenate([np.asarray(part["defined"], dtype=bool) for part in parts]),
    )


def _ordered(origins):
    _require(origins, "El informe necesita al menos un origen")
    return sorted(origins, key=lambda report: report["request"]["origin"]["year"])


def _lengths(item):
    """Longitudes de todas las series por sesión de un objetivo y un modelo."""
    pairs = (item["session_ic"], item["session_rank_ic"], *item["importance"].values())
    single = (item["session_mae_raw"], item["session_mae_calibrated"], item["session_mae_zero"])
    return {len(part) for pair in pairs for part in pair} | {len(values) for values in single}


def pooled(origins):
    """Series de la evaluación de los orígenes unidas en orden temporal, sin solaparse."""
    origins = _ordered(origins)
    times = np.concatenate([np.asarray(r["sessions"]["time"], dtype=np.int64) for r in origins])
    _require(np.all(np.diff(times) > 0), "Las sesiones de los orígenes se solapan o se repiten")
    day = np.concatenate([np.asarray(r["sessions"]["day"], dtype=np.int64) for r in origins])
    series = {}
    for target in TARGETS:
        zero = np.concatenate([r["targets"][target]["zero_mae"] for r in origins])
        for model in MODELS:
            parts = [r["targets"][target][model] for r in origins]
            series[target, model] = dict(
                session_ic=_joined([part["session_ic"] for part in parts]),
                session_rank_ic=_joined([part["session_rank_ic"] for part in parts]),
                session_mae_raw=np.concatenate([part["session_mae_raw"] for part in parts]),
                session_mae_calibrated=np.concatenate(
                    [part["session_mae_calibrated"] for part in parts]
                ),
                session_mae_zero=zero,
                importance={
                    family: _joined([part["importance"][family] for part in parts])
                    for family in parts[0]["importance"]
                },
                validation_ic=[part["validation_ic"] for part in parts],
            )
    _require(
        all(_lengths(item) == {len(times)} for item in series.values()),
        "Las series no tienen una sesión por cada sesión de los orígenes",
    )
    return times, day, series


def _member(name, values, defined, direction):
    return dict(name=name, values=values, defined=defined, direction=direction)


def _mean(values, defined):
    return float(values[defined].mean()) if defined.any() else None


def _summary(item):
    ic, rank = item["session_ic"], item["session_rank_ic"]
    defined = ic[0][ic[1]]
    spread = float(np.std(defined, ddof=1)) if len(defined) > 2 else 0.0
    calibrated, zero = item["session_mae_calibrated"], item["session_mae_zero"]
    return dict(
        mean_session_ic=_mean(*ic),
        mean_session_rank_ic=_mean(*rank),
        session_icir=float(defined.mean() / spread) if spread > 0 else None,
        defined_ic_sessions=int(ic[1].sum()),
        mean_session_mae_raw=float(np.mean(item["session_mae_raw"])),
        mean_session_mae_calibrated=float(np.mean(calibrated)),
        mean_session_mae_zero=float(np.mean(zero)),
        mean_session_mae_calibrated_minus_zero=float(np.mean(calibrated - zero)),
        validation_ic_by_origin=item["validation_ic"],
    )


def _ceiling(series):
    """Modelo de cada objetivo con mayor IC medio de validación de los orígenes."""
    result = {}
    for target in TARGETS:
        scores = {}
        for model in MODELS:
            values = series[target, model]["validation_ic"]
            _require(all(v is not None for v in values), "Un origen no define el IC de validación")
            scores[model] = float(np.mean(values))
        # `max` se queda con el primero entre iguales, Ridge.
        chosen = max(MODELS, key=lambda model: scores[model])
        result[target] = dict(model=chosen, validation_ic=scores)
    return result


def report(origins, inference):
    """Medias, familias con intervalos por bloques, importancias y techo de cada objetivo."""
    times, day, series = pooled(origins)
    main, sensitivity = inference["block_days"], inference["sensitivity_block_days"]

    def family(members, blocks=(main,)):
        return {
            str(length): evaluate_family(members, day, block_length=length, inference=inference)
            for length in blocks
        }

    labels = [(target, model) for target in TARGETS for model in MODELS]
    ic = [_member("/".join(k), *series[k]["session_ic"], "greater") for k in labels]
    rank = [_member("/".join(k), *series[k]["session_rank_ic"], "greater") for k in labels]
    mae = [
        _member(
            "/".join(k),
            series[k]["session_mae_calibrated"] - series[k]["session_mae_zero"],
            np.ones(len(times), dtype=bool),
            "less",
        )
        for k in labels
    ]
    importance = {
        "/".join(k): family(
            [
                _member(name, values, defined, "greater")
                for name, (values, defined) in series[k]["importance"].items()
            ]
        )
        for k in labels
    }
    return dict(
        schema_version=1,
        kind=KIND,
        origins=[r["request"]["origin"] for r in _ordered(origins)],
        sessions=len(times),
        days=len(np.unique(day)),
        summaries={"/".join(k): _summary(series[k]) for k in labels},
        families=dict(
            session_ic=family(ic, (main, *sensitivity)),
            session_rank_ic=family(rank),
            session_mae_calibrated_minus_zero=family(mae),
        ),
        importance=importance,
        ceiling=_ceiling(series),
        inference=inference,
    )
