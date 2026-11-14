"""Bootstrap circular por origen, con emparejamiento de todos los contrastes de una familia."""

import numpy as np
from statsmodels.stats.multitest import multipletests

from .paired_comparisons import circular_block_counts, family_intervals
from .variant_screening_analysis import stepdown

_CHUNK = 256


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def stratified_draws(values, days, origins, *, block_length, replicates, seed):
    """Remuestrear días dentro de cada origen, sin unir por un bloque años disjuntos.

    Cada origen conserva su peso original en la media por sesión. Todos los contrastes
    reciben los mismos sorteos. Los generadores son locales y se separan por origen.
    """
    values = np.asarray(values, dtype=np.float64)
    days, origins = np.asarray(days), np.asarray(origins)
    _require(
        values.ndim == 2 and 1 <= values.shape[1] <= 64, "La familia necesita columnas acotadas"
    )
    _require(
        days.shape == origins.shape == (len(values),) and len(values) > 0,
        "Los estratos no concilian con las sesiones",
    )
    _require(np.isfinite(values).all(), "Las métricas no son finitas")
    _require(
        type(replicates) is int and 32 <= replicates <= 100_000, "Las réplicas deben estar acotadas"
    )
    _require(type(block_length) is int and block_length >= 1, "La longitud del bloque no es válida")
    _require(type(seed) is int and 0 <= seed < 2**32, "La semilla del bootstrap no es válida")
    draws = np.zeros((replicates, values.shape[1]))
    for origin in np.unique(origins):
        rows = origins == origin
        _, period = np.unique(days[rows], return_inverse=True)
        periods = int(period.max()) + 1
        _require(block_length < periods, "Se necesitan más días que el bloque en cada origen")
        sizes = np.bincount(period, minlength=periods)
        sums = np.column_stack(
            [
                np.bincount(period, weights=values[rows, column], minlength=periods)
                for column in range(values.shape[1])
            ]
        )
        weight = int(rows.sum()) / len(values)
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(origin)]))
        for start in range(0, replicates, _CHUNK):
            size = min(_CHUNK, replicates - start)
            counts = circular_block_counts(rng, size, periods, block_length)
            draws[start : start + size] += weight * (counts @ sums) / (counts @ sizes)[:, None]
    return draws


def compare_family(members, days, origins, *, block_length, settings):
    """Intervalos y valores p de una familia sobre su intersección de sesiones definidas."""
    _require(1 <= len(members) <= 64, "La familia de contrastes debe estar acotada")
    days, origins = np.asarray(days), np.asarray(origins)
    _require(days.shape == origins.shape and days.ndim == 1, "Las sesiones y orígenes no concilian")
    for member in members:
        _require(
            isinstance(member["values"], np.ndarray)
            and member["values"].shape == member["defined"].shape == days.shape
            and member["defined"].dtype == np.bool_
            and member["direction"] in ("less", "greater")
            and np.isfinite(member["values"]).all(),
            "Un contraste no conserva valores, máscara y dirección válidos",
        )
    defined = np.logical_and.reduce([member["defined"] for member in members])
    base = dict(
        block_days=block_length,
        sessions=int(defined.sum()),
        days=len(np.unique(days[defined])),
        stratification="origin",
        available=False,
        members=[],
    )
    if not defined.any():
        return dict(base, reason="La familia no tiene sesiones comunes definidas")
    selected_days, selected_origins = days[defined], origins[defined]
    if any(
        len(np.unique(selected_days[selected_origins == y])) <= block_length
        for y in np.unique(selected_origins)
    ):
        return dict(base, reason="Se necesitan más días que el bloque en cada origen")
    values = np.column_stack([member["values"][defined] for member in members])
    draws = stratified_draws(
        values,
        selected_days,
        selected_origins,
        block_length=block_length,
        replicates=settings["replicates"],
        seed=settings["seed"],
    )
    estimate = values.mean(axis=0)
    tests = stepdown(estimate, draws)
    bounds = family_intervals(estimate, draws, settings["confidence"])
    holm = multipletests(tests["marginal"], method="holm")[1]
    alpha = 1 - settings["confidence"]
    rows = []
    for index, member in enumerate(members):
        favourable = estimate[index] < 0 if member["direction"] == "less" else estimate[index] > 0
        rows.append(
            dict(
                name=member["name"],
                estimate=float(estimate[index]),
                interval=[float(bounds["lower"][index]), float(bounds["upper"][index])],
                simultaneous_interval=[
                    float(bounds["joint_lower"][index]),
                    float(bounds["joint_upper"][index]),
                ],
                bootstrap_std=float(tests["spread"][index]),
                marginal_pvalue=float(tests["marginal"][index]),
                romano_wolf_pvalue=float(tests["stepdown"][index]),
                holm_pvalue=float(holm[index]),
                favourable=bool(favourable),
                passes_romano_wolf=bool(favourable and tests["stepdown"][index] <= alpha),
                passes_holm=bool(favourable and holm[index] <= alpha),
                degenerate_bootstrap=bool(tests["spread"][index] == 0),
            )
        )
    return dict(base, available=True, reason=None, members=rows, critical=bounds["critical"])
