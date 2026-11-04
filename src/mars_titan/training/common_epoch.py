"""Época común de la criba (#746) dentro de los entrenadores cronológicos.

Con la parada `common_epoch` de `selection`, cada validación completa de un ajuste se puntúa
también con el IC de Pearson por sesión de sus propias filas, el estado de esa época queda
fijado en los puntos de control y el ajuste espera al grupo de semillas tras la última época.
El IC se calcula con la función del análisis de la criba
(`scale_diagnostics.session_correlations`) sobre las filas que ya emite la validación del
entrenador, conciliadas con la vista, así que no repite ninguna inferencia.

La ventana construye `EpochValidation` y el entrenador lo usa en cada validación. Al recibir
la época común, `chosen_state` marca como elegido su estado fijado y devuelve los campos del
informe. La ventana emite después sus tramos con ese estado y, con `versions`, la calibración
y la evaluación de otras épocas fijadas, que usa el árbitro de versiones (#789) y la media de
semillas cuando su época combinada es otra. Los estados fijados son transitorios y la ventana
los libera tras confirmar todas sus predicciones.
"""

import pyarrow as pa

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.storage import sha256
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import ForecastPanel

from .checkpoints import choose_pinned
from .selection import COMMON_EPOCH

# Columnas de la predicción puntual de cada época, el contrato mínimo de `ForecastPanel`.
POINT_COLUMNS = ("sample_id", "market", "prediction_at", "target", "prediction")
# Tramos que emite cada versión. La validación de cada época ya la puntúa el propio ajuste.
VERSIONED = ("calibration", "evaluation")


def checked_scoring(scoring, rule):
    """Validar la puntuación por época, que exige la época común y solo existe con ella.

    `min_assets` es el mínimo de activos del IC por sesión del contraste principal y
    `predictions` dice si cada época guarda su predicción puntual de validación.
    """
    common = rule.get("stopping") == COMMON_EPOCH
    if scoring is None:
        if common:
            raise ValueError("La época común necesita declarar cómo se puntúa cada validación")
        return None
    if (
        not common
        or not isinstance(scoring, dict)
        or set(scoring) != {"min_assets", "predictions"}
        or type(scoring["min_assets"]) is not int
        or scoring["min_assets"] < 3
        or type(scoring["predictions"]) is not bool
    ):
        raise ValueError(
            "La puntuación por época solo existe con la época común y declara los activos "
            "mínimos del IC y si guarda las predicciones"
        )
    return dict(scoring)


def bind_versions(report, versions, *, epochs, common):
    """Fijar las épocas adicionales que emite la ventana. Una reanudación no puede cambiarlas.

    Las versiones son épocas fijadas distintas de la común, en orden y sin repetir. Sin
    versiones la ventana solo emite el estado elegido.
    """
    epochs_list = [] if versions is None else list(versions)
    if (
        any(type(epoch) is not int or not 1 <= epoch <= epochs for epoch in epochs_list)
        or epochs_list != sorted(set(epochs_list))
        or common in epochs_list
    ):
        raise ValueError(
            "Las versiones son épocas evaluadas distintas de la común, en orden y sin repetir"
        )
    recorded = report.get("versions")
    if recorded is not None and recorded["epochs"] != epochs_list:
        raise ValueError("La ventana ya recibió otras versiones")
    if recorded is None:
        report["versions"] = dict(epochs=epochs_list, predictions={})
    return report["versions"]


def version_path(output, epoch, partition):
    return output / "versions" / f"epoch-{epoch:04d}-{partition}-predictions.parquet"


class EpochValidation:
    """Puntuar con el IC por sesión cada validación completa de un ajuste.

    `rows()` crea el destino que recibe las filas de la validación del entrenador y `tables`
    lo convierte en las tablas conciliadas con la vista, las mismas que se confirman al
    predecir un tramo. Con `directory`, cada época guarda además su predicción puntual, que
    necesitan las salidas que combinan semillas.
    """

    def __init__(self, rows, tables, *, min_assets, directory=None):
        if type(min_assets) is not int or min_assets < 3:
            raise ValueError("El IC por sesión necesita al menos tres activos")
        self._rows, self._tables, self.min_assets = rows, tables, min_assets
        self.directory = directory

    def rows(self):
        return self._rows()

    def score(self, epoch, rows, metrics):
        """Campos de la validación de `epoch` que añade la época común."""
        table = pa.concat_tables(self._tables(rows, metrics))
        panel = ForecastPanel.from_arrow(table)
        values, defined = sd.session_correlations(
            panel, panel.prediction, min_assets=self.min_assets
        )
        record = dict(
            session_ic=float(values[defined].mean()) if defined.any() else None,
            session_ic_values=values.tolist(),
            session_ic_defined=defined.tolist(),
            # El MAE por sesión permite repetir la regla anterior en la comprobación de #746.
            session_mae_values=sd.session_mae(panel, panel.prediction).tolist(),
            session_time=panel.session_time.tolist(),
            cohort_sha256=panel.cohort_sha256,
        )
        if self.directory is not None:
            path = self.directory / f"validation-epoch-{epoch:04d}.parquet"
            written = atomic_parquet_batches(path, [table.select(POINT_COLUMNS)])
            record["epoch_predictions"] = dict(
                path=f"{self.directory.name}/{path.name}", sha256=sha256(path), rows=written
            )
        return record


def chosen_state(checkpoints, identity, report, history, selection, epoch):
    """Campos del informe con el estado de la época común marcado como elegido.

    `best_*` describe el estado emitido, que es el de la época común, y `best_score` es su
    MAE de validación para que la ventana pueda comprobarlo al repetir la validación. El mejor
    estado por MAE del propio ajuste queda en `mae_selection` como sensibilidad.
    """
    reference = report.get("epoch_states", {}).get(str(epoch))
    entry = next((item for item in history if item["epoch"] == epoch), None)
    if reference is None or entry is None:
        raise ValueError("La época común no tiene un estado fijado ni una validación completa")
    choose_pinned(checkpoints, expected_identity=identity, expected_sha256=reference["sha256"])
    return dict(
        best_checkpoint=dict(reference),
        best_epoch=epoch,
        best_score=entry["validation"]["session_mae"],
        common_epoch=epoch,
        common_epoch_session_ic=entry["validation"]["session_ic"],
        mae_selection=dict(best_epoch=selection["best_epoch"], best_score=selection["best_score"]),
    )
