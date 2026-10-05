"""Vistas de información posteriores a la resolución temporal del corpus admitido."""

import copy
import csv
from pathlib import Path

from mars_titan.data.company_factors import FACTOR_DEFINITIONS
from mars_titan.data.information_views import InformationView, _digest
from mars_titan.data.macro import _catalog
from mars_titan.data.storage import sha256
from mars_titan.models.baselines.inputs import MODALITIES

from .cohort_contract import representation_identity


def _node(name, source, block, value, *, mask=(), age=(), dependencies=(), transformation, history):
    return dict(
        name=name,
        source=source,
        block=block,
        value=list(value),
        mask=list(mask),
        age=list(age),
        dependencies=sorted(dependencies),
        transformation=transformation,
        history=history,
    )


def _numeric_nodes(block, names, dependencies, histories):
    nodes, width = [], len(names)
    all_names = set(names)
    pending = list(names)
    while pending:
        name = pending.pop()
        for dependency in dependencies.get(name, ()):
            if dependency not in all_names:
                all_names.add(dependency)
                pending.append(dependency)
    positions = {name: position for position, name in enumerate(names)}
    for name in [*names, *sorted(all_names - set(names))]:
        index = positions.get(name)
        nodes.append(
            _node(
                f"{block}/{name}",
                block,
                block if index is not None else None,
                [index] if index is not None else [],
                mask=[width + index] if index is not None else [],
                age=[2 * width + index] if index is not None else [],
                dependencies=[f"{block}/{key}" for key in dependencies.get(name, ())],
                transformation="numeric_context/signed_log1p_presence_log1p_age",
                history=histories.get(name, "Última publicación disponible, con unidad y revisión"),
            )
        )
    return nodes


def corpus_view(dataset, *, macro_catalog):
    """Derivar el catálogo de la representación, ratios y fórmulas macro existentes."""
    representation = representation_identity(dataset.manifest.get("representation", {}))
    concepts, indicators = (
        representation["fundamental_concepts"],
        representation["macro_indicators"],
    )
    if (
        representation["context_sessions"] != dataset.context
        or not 1 <= len(concepts) <= 512
        or not 1 <= len(indicators) <= 1024
    ):
        raise ValueError("La representación no conserva sus dimensiones semánticas")
    if dataset.temporal is not None and indicators != dataset.temporal.macro.indicators:
        raise ValueError("El catálogo no identifica los valores macro efectivos de TemporalInputs")
    catalog = Path(macro_catalog)
    if catalog.is_symlink() or not catalog.is_file() or catalog.stat().st_size > 2 * 1024**2:
        raise ValueError("El catálogo macro excede su presupuesto o no es regular")
    catalog_hash = sha256(catalog)
    with catalog.open() as stream:
        entries, dependencies, _, _ = _catalog(list(csv.DictReader(stream)))
    if sha256(catalog) != catalog_hash or not set(indicators) <= entries.keys():
        raise ValueError("El catálogo cambió o falta una variable de la representación")
    variables = []
    for column, name in enumerate(("open", "high", "low", "close", "volume")):
        variables.append(
            _node(
                f"prices/{name}",
                "prices",
                "prices",
                range(column, dataset.context * 5, 5),
                dependencies=["prices/close"] if name in {"open", "high", "low"} else [],
                transformation="log_relative_volume"
                if name == "volume"
                else "log_relative_first_close",
                history=f"{dataset.context} sesiones de precios disponibles",
            )
        )
    variables.extend(
        [
            _node(
                "news",
                "news",
                "news",
                range(384),
                transformation=representation["text_aggregation"],
                history=f"{representation['news_lookback_sessions']} sesiones disponibles",
            ),
            _node(
                "charts",
                "charts",
                "charts",
                range(512),
                dependencies=[f"prices/{k}" for k in ("open", "high", "low", "close", "volume")],
                transformation="resnet18_224_rgb_imagenet_no_crop",
                history=f"{dataset.context} sesiones de precios disponibles",
            ),
        ]
    )
    factors = {
        f"company:{name}:ratio": {f"us-gaap:{tag}:USD" for tag, _ in terms}
        | {f"us-gaap:{denominator}:USD"}
        for name, terms, denominator in FACTOR_DEFINITIONS
    }
    if any(name.startswith("company:") and name not in factors for name in concepts):
        raise ValueError("Falta la definición de dependencias de un ratio contable")
    variables.extend(
        _numeric_nodes(
            "fundamentals",
            concepts,
            factors,
            {key: "Componentes de la misma presentación, periodo y unidad USD" for key in factors},
        )
    )
    variables.extend(
        _numeric_nodes(
            "macro",
            indicators,
            dependencies,
            {
                key: f"{entry['formula'] or 'Nivel publicado'}, frecuencia {entry['frequency']}, "
                f"política {entry['vintage_policy']}"
                for key, entry in entries.items()
            },
        )
    )
    return InformationView(
        dict(
            schema_version=1,
            kind="information_view",
            cohort_sha256=dataset.identity,
            representation_sha256=_digest(
                dict(
                    representation=representation,
                    temporal_view=dataset.manifest.get("temporal_view"),
                    macro_catalog_sha256=catalog_hash,
                )
            ),
            shapes=dict(
                prices=[dataset.context, 5],
                news=[384],
                charts=[512],
                fundamentals=[3 * len(concepts)],
                macro=[3 * len(indicators)],
            ),
            variables=variables,
            allowed_sources=list(MODALITIES),
            allowed_variables=[row["name"] for row in variables],
        )
    )


class ViewDataset:
    """Aplicar una vista a los lotes finales sin materializar otra edición del corpus."""

    def __init__(self, source, view):
        if source.identity != view.manifest["cohort_sha256"]:
            raise ValueError("La vista no pertenece a la misma cohorte de origen")
        self.source, self.view = source, view
        self.counts = copy.deepcopy(source.manifest["counts"])
        self.partitions = source.partitions
        self.identity = view.identity

    def batches(self, *, cursor=None, **kwargs):
        original = None
        if cursor is not None:
            if (
                not isinstance(cursor, dict)
                or set(cursor) != {"information_view_sha256", "source_cursor"}
                or cursor["information_view_sha256"] != self.identity
                or not isinstance(cursor["source_cursor"], dict)
            ):
                raise ValueError("El cursor corresponde a otra vista")
            original = cursor["source_cursor"]
        for batch in self.source.batches(cursor=original, **kwargs):
            result = self.view.apply(batch)
            result["confirmed_cursor"] = dict(
                information_view_sha256=self.identity, source_cursor=result["confirmed_cursor"]
            )
            yield result
