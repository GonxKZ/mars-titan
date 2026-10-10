"""Episodios remuestreados o sintéticos del postentrenamiento emparejado anterior.

Pertenecen al diseño de #128 que compara la condición real con `real_resampled` y
`real_synthetic`. Se conservan por trazabilidad. La etapa de adaptadores de la campaña
con máscaras no importa este módulo: lee solo cohortes reales con `PairedInputs`.
"""

from dataclasses import asdict

from mars_titan.episodes.augmentation import training_visits
from mars_titan.episodes.windows import EpisodeView

from .inputs import PairedInputs


class AugmentedInputs(PairedInputs):
    """Compartir la fuente real y cargar un único episodio adicional cada vez."""

    def __init__(
        self, train, validation, parent, *, windows=(), synthetic=None, synthetic_identity=None
    ):
        self.windows, self.synthetic = tuple(windows), synthetic
        self._synthetic_identity = synthetic_identity
        super().__init__(train, validation, parent)

    def _episodes(self, train):
        windows, synthetic = self.windows, self.synthetic
        if self.masked and (
            windows or synthetic is not None or self._synthetic_identity is not None
        ):
            # Los episodios sintéticos no tienen bits de presencia ni causas de ausencia.
            raise ValueError("La edición con máscaras solo admite la condición real")
        if len(windows) > 100_000:
            raise ValueError("El número de episodios excede el presupuesto")
        for window in windows:
            EpisodeView(train, window)
        return [asdict(window) for window in windows], self._synthetic_identity

    def _extra_visits(self, condition, epoch, seed):
        if not self.windows:
            raise ValueError("La condición requiere un aumento confirmado")
        return training_visits(self.train, self.windows, epoch=epoch, seed=seed)

    def _episode(self, source, episode, condition):
        window = self.windows[episode]
        if condition == "real_resampled":
            view = EpisodeView(source, window)
        else:
            if self.synthetic is None:
                raise ValueError("Faltan episodios sintéticos emparejados")
            view = self.synthetic(episode)
            if (self.identity["synthetic"] or {}).get(str(episode)) != view.source.manifest_sha256:
                raise ValueError("El episodio sintético no conserva su identidad")
            counts = [source.index[i][1] for i in range(window.decision_start, window.stop)]
            other = view.window
            observed = [view.source.index[i][1] for i in range(other.decision_start, other.stop)]
            if (
                view.shapes != self.shapes
                or view.partition != "train"
                or counts != observed
                or view.window.origin != "synthetic"
            ):
                raise ValueError("El episodio sintético no conserva el contrato emparejado")
        return view
