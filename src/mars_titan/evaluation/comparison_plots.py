"""Gráficas exportables a partir del informe numérico, sin volver a ejecutar predictores."""

from pathlib import Path

import numpy as np


def write_plots(directory, report, curves):
    """Escribir PNG y SVG. Las medidas ausentes no se dibujan como ceros."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    created = []

    def save(figure, name):
        for suffix in ("png", "svg"):
            path = directory / f"{name}.{suffix}"
            metadata = (
                {"Creator": "MARS-TITAN", "Date": None}
                if suffix == "svg"
                else {"Software": "Matplotlib"}
            )
            figure.savefig(path, dpi=160, metadata=metadata)
            created.append(f"plots/{path.name}")
        plt.close(figure)

    names = [name for name, result in report["models"].items() if result["pooled"]]
    width = max(8, min(18, len(names) * 0.55))
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(width, 5), layout="constrained")
    for offset, field, label in (
        (-0.2, "mae_raw", "MAE bruto"),
        (0.2, "mae_calibrated", "MAE calibrado"),
    ):
        values = [report["models"][name]["pooled"].get(field) for name in names]
        axis.bar(
            x + offset,
            [np.nan if value is None else value for value in values],
            width=0.4,
            label=label,
        )
    if names:
        axis.axhline(
            report["models"][names[0]]["pooled"]["zero_mae"],
            color="black",
            linestyle="--",
            label="Predicción cero",
        )
        axis.set_xticks(x, names, rotation=90)
        axis.legend()
    else:
        axis.text(
            0.5,
            0.5,
            "No hay modelos con todas las semillas y orígenes",
            ha="center",
            transform=axis.transAxes,
        )
    axis.set(ylabel="MAE medio por sesión", title="Error sobre las mismas filas")
    save(figure, "mae")

    figure, axis = plt.subplots(figsize=(width, 5), layout="constrained")
    for offset, field, label in ((-0.2, "pearson_ic", "Pearson"), (0.2, "rank_ic", "Spearman")):
        values = [report["models"][name]["pooled"].get(field) for name in names]
        axis.bar(
            x + offset,
            [np.nan if value is None else value for value in values],
            width=0.4,
            label=label,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, names, rotation=90)
    axis.set(ylabel="Correlación media por sesión", title="Ordenación de los activos")
    axis.legend()
    save(figure, "information-coefficient")

    for group, comparison in report["comparisons"].items():
        family = comparison["metrics"]["mae_raw"].get("primary", {})
        result = family.get("main", {})
        if not result.get("available"):
            continue
        members = result["members"]
        figure, axis = plt.subplots(figsize=(10, max(3, len(members) * 0.6)), layout="constrained")
        for position, member in enumerate(members):
            lower, upper = member["simultaneous_interval"]
            axis.plot([lower, upper], [position, position], color="tab:blue", linewidth=2)
            axis.scatter([member["estimate"]], [position], color="black", s=25)
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_yticks(range(len(members)), [m["name"] for m in members])
        axis.set(
            xlabel="Diferencia de MAE, variante menos control",
            title=f"{group}: intervalos simultáneos de la familia principal",
        )
        save(figure, f"contrasts-{group}")

    figure, axis = plt.subplots(figsize=(width, 5), layout="constrained")
    resource_names = list(report["models"])
    values = []
    for name in resource_names:
        observed = [row["wall_seconds"] for row in report["resources"] if row["model"] == name]
        values.append(
            sum(observed) / 3600
            if observed and all(value is not None for value in observed)
            else np.nan
        )
    axis.bar(np.arange(len(resource_names)), values)
    axis.set_xticks(np.arange(len(resource_names)), resource_names, rotation=90)
    axis.set(
        ylabel="Horas registradas, incluidos intentos fallidos",
        title="Coste incremental observado. El ajuste del padre figura por separado",
    )
    save(figure, "resources")

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    any_interval = False
    for name in names:
        rows = [row for row in report["per_seed"] if row["model"] == name]
        for nominal in (0.8, 0.95):
            coverage = []
            for row in rows:
                quantiles = row["partitions"]["evaluation"]["raw"].get("quantiles")
                for item in (quantiles or {}).get("intervals", []):
                    if abs(item["nominal"] - nominal) < 1e-12:
                        coverage.append(item["coverage"])
            if coverage:
                axis.scatter(
                    [nominal], [np.mean(coverage)], label=name if nominal == 0.8 else None, s=25
                )
                any_interval = True
    axis.plot([0, 1], [0, 1], color="black", linestyle="--")
    axis.set(
        xlabel="Cobertura nominal",
        ylabel="Cobertura observada media",
        title="Intervalos originales de los modelos",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    if any_interval and len(names) <= 10:
        axis.legend(fontsize=8)
    save(figure, "coverage")

    for name in report["models"]:
        selected = [row for row in curves if row["model"] == name]
        if not selected:
            continue
        figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
        keys = sorted({(row["origin"], str(row["seed"])) for row in selected})
        axis_name = selected[0]["axis"]
        field = "validation_ic" if axis_name == "round" else "validation_mae"
        for year, seed in keys:
            points = sorted(
                (row for row in selected if row["origin"] == year and str(row["seed"]) == seed),
                key=lambda row: row["step"],
            )
            axis.plot(
                [row["step"] for row in points],
                [np.nan if row.get(field) is None else row[field] for row in points],
                label=f"{year}, semilla {seed}",
                linewidth=1,
            )
            picked = [row for row in points if row.get("selected") and row.get(field) is not None]
            axis.scatter(
                [row["step"] for row in picked], [row[field] for row in picked], marker="x", s=35
            )
        if axis_name == "alpha":
            axis.set_xscale("log")
        axis.set(
            xlabel={"epoch": "Época", "round": "Ronda", "alpha": "Alpha"}[axis_name],
            ylabel="IC de validación" if field == "validation_ic" else "MAE de validación",
            title=f"Validación de {name}",
        )
        if len(keys) <= 8:
            axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        save(figure, f"learning-{name}")
    return created
