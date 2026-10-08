"""Contratos temporales por mercado, sin abrir las fuentes declaradas."""

from mars_titan.data.input_policy import STRICT_INPUTS, masked_inputs
from mars_titan.evaluation.splits import build_folds

from .cohort_contract import input_identity


def validate_temporal_view(view, *, input_policy=STRICT_INPUTS):
    """Separar el contexto completo estricto de las entradas históricas del padre."""
    masked = masked_inputs(input_policy)
    common = {"schema_version", "protocol", "fold", "parent_manifest", "parent_sha256"}
    fields = (
        {"input_policy", "mask_contract", "selection_partition", "recover_annual_boundaries"}
        if masked
        else {"macro_path", "macro_sha256", "admission_path", "admission_sha256"}
    )
    if (
        not isinstance(view, dict)
        or set(view) != common | fields
        or type(view["schema_version"]) is not int
        or view["schema_version"] != (2 if masked else 1)
    ):
        raise ValueError("La vista temporal no cumple su contrato")
    input_identity(view, input_policy=input_policy)
    protocol = view["protocol"]
    if view["fold"] not in build_folds(protocol):
        raise ValueError("La ventana no pertenece al protocolo declarado")
    if masked and (
        type(protocol["schema_version"]) is not int
        or protocol["train_start"] != "2000-01-01"
        or protocol["final_test_start"] != "2024-01-01"
        or view["selection_partition"] != "validation"
        or view["recover_annual_boundaries"] is not True
    ):
        raise ValueError(
            "El contrato histórico no conserva el inicio, la selección o la recuperación"
        )
    return masked


def temporal_contracts(manifest, *, input_policy=STRICT_INPUTS):
    """Validar una vista por mercado o la unión explícita de US y CN."""
    single, joint = "temporal_view" in manifest, "temporal_views" in manifest
    if single and joint:
        raise ValueError("Una supervisión no puede declarar dos formatos temporales")
    if not (single or joint):
        return {}
    input_identity(manifest, input_policy=input_policy)
    if single:
        view = manifest["temporal_view"]
        if not isinstance(view, dict) or not isinstance(view.get("protocol"), dict):
            raise ValueError("Falta el contrato temporal de la ventana")
        market = view["protocol"].get("market")
        if market not in ("US", "CN"):
            raise ValueError("El contrato temporal necesita un mercado US o CN")
        validate_temporal_view(view, input_policy=input_policy)
        return {market: view}
    views = manifest["temporal_views"]
    if not isinstance(views, dict) or set(views) != {"US", "CN"}:
        raise ValueError("La vista conjunta requiere exactamente los calendarios US y CN")
    protocol, fold = None, None
    for market, view in views.items():
        masked = validate_temporal_view(view, input_policy=input_policy)
        if (
            not isinstance(view, dict)
            or type(view.get("schema_version")) is not int
            or view["schema_version"] != (2 if masked else 1)
            or not isinstance(view.get("protocol"), dict)
            or view["protocol"].get("market") != market
            or view["protocol"].get("final_test_start") != "2024-01-01"
            or view.get("fold") not in build_folds(view["protocol"])
        ):
            raise ValueError("Un mercado no conserva su ventana y la reserva final")
        common = {key: value for key, value in view["protocol"].items() if key != "market"}
        if protocol is not None and (common != protocol or view["fold"] != fold):
            raise ValueError("Los mercados no comparten cortes, semillas y margen temporal")
        protocol, fold = common, view["fold"]
    assets = manifest.get("assets")
    if (
        manifest.get("final_test_opened") is not False
        or not isinstance(assets, list)
        or any(
            not isinstance(asset, dict) or asset.get("market") not in ("US", "CN")
            for asset in assets
        )
        or {asset.get("market") for asset in assets} != set(views)
        or manifest.get("markets") not in (["CN", "US"], ["US", "CN"])
    ):
        raise ValueError("La población conjunta no corresponde a sus dos calendarios")
    return dict(views)


def temporal_fold(manifest, *, input_policy=STRICT_INPUTS):
    """Obtener los límites mensuales comunes después de validar cada mercado."""
    contracts = temporal_contracts(manifest, input_policy=input_policy)
    if not contracts:
        raise ValueError("Falta una ventana temporal declarada")
    return next(iter(contracts.values()))["fold"]
