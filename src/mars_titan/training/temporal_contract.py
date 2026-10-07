"""Contratos temporales por mercado, sin abrir las fuentes declaradas."""

from mars_titan.evaluation.splits import build_folds


def temporal_contracts(manifest):
    """Normalizar la vista histórica o la unión explícita de US y CN."""
    single, joint = "temporal_view" in manifest, "temporal_views" in manifest
    if single and joint:
        raise ValueError("Una supervisión no puede declarar dos formatos temporales")
    if not (single or joint):
        return {}
    if single:
        view = manifest["temporal_view"]
        if not isinstance(view, dict) or not isinstance(view.get("protocol"), dict):
            raise ValueError("Falta el contrato temporal de la ventana")
        market = view["protocol"].get("market")
        if market not in ("US", "CN"):
            raise ValueError("El contrato temporal necesita un mercado US o CN")
        return {market: view}
    views = manifest["temporal_views"]
    if not isinstance(views, dict) or set(views) != {"US", "CN"}:
        raise ValueError("La vista conjunta requiere exactamente los calendarios US y CN")
    protocol, fold = None, None
    for market, view in views.items():
        if (
            not isinstance(view, dict)
            or type(view.get("schema_version")) is not int
            or view["schema_version"] != 1
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


def temporal_fold(manifest):
    """Obtener los límites mensuales comunes después de validar cada mercado."""
    contracts = temporal_contracts(manifest)
    if not contracts:
        raise ValueError("Falta una ventana temporal declarada")
    return next(iter(contracts.values()))["fold"]
