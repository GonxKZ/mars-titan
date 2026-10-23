"""Ajuste protegido sobre pérdidas maduras de fases cerradas o prefijos nativos."""

import hashlib
from dataclasses import asdict
from pathlib import Path

import torch

from mars_titan.memory.financial_session import FinancialSession, _check_callbacks
from mars_titan.memory.mechanism_collection_record import CollectedMechanisms
from mars_titan.memory.mechanism_supervision import MechanismSupervision
from mars_titan.models.titans.mechanism_catalog import MechanismCatalog
from mars_titan.models.titans.mechanism_policy import (
    ACTIONS,
    FEATURES,
    POINT_ACTIONS,
    MechanismPolicy,
)
from mars_titan.models.titans.mechanism_selection import digest, policy_signature
from mars_titan.models.titans.state import check_finite
from mars_titan.training.learning_hold import require_learning_allowed

from . import mechanism_fit_contract, mechanism_fit_window
from .mechanism_fit_contract import MechanismFitConfig


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cohorts(sources, *, config, device):
    require_learning_allowed("lectura de señales para la política de mecanismos")
    if config.source_mode == "native_windows":
        return _window_cohorts(sources, config=config, device=device)
    _require(
        type(sources) in (list, tuple) and 1 <= len(sources) <= config.max_sources,
        "El ajuste necesita sesiones fuente explícitas y acotadas",
    )
    records, used, cohort_keys, source_keys, budgets = [], [], set(), set(), [0, 0]
    for source in sources:
        _require(
            type(source) is FinancialSession and source._collection_config is not None,
            "La fuente debe ser una sesión financiera con recogida identificada",
        )
        _check_callbacks(source)
        source.consumer.verify()
        config.check_catalog(source.consumer.catalog)
        before = source.snapshot()
        _require(
            before["phase_closed"]
            and source.consumer.model_id == config.parent_consumer_id
            and config.fit_start <= source.phase.decision_start
            and source.phase.close_at < config.fit_end,
            "La fuente no está cerrada dentro del tramo del padre declarado",
        )
        source_key = source.contract_id, before["record_sha256"]
        _require(source_key not in source_keys, "La sesión fuente se repite")
        source_keys.add(source_key)
        bundle = source._bundle(before["state"])
        _require(bundle is not None, "La sesión no tiene generación confirmada")
        generations = bundle["mechanism_collections"]["entries"]
        record = dict(
            contract_id=source.contract_id,
            model_id=source.model_id,
            native_record_sha256=before["record_sha256"],
            phase=asdict(source.phase),
            generation=before["cursor"],
            collections=[],
        )
        for entry in generations:
            budgets[0] += entry["packet"]["bytes"] + entry["receipt"]["bytes"]
            _require(
                budgets[0] <= config.max_source_bytes, "Las fuentes superan sus bytes declarados"
            )
            receipt = source.mechanism_collection(entry["generation"])
            packet = CollectedMechanisms.restore(receipt["collection"])
            body = packet.payload()
            config.check_catalog(MechanismCatalog.restore(body["catalog"]))
            _require(
                body["catalog"] == source.consumer.catalog.identity(),
                "La colección cambia el catálogo del consumidor padre",
            )
            observation = body["observation"]["observation"]
            _require(
                observation["consumer_id"] == config.parent_consumer_id
                and config.fit_start <= entry["cutoff"] < config.fit_end,
                "La colección pertenece a otro padre o tramo",
            )
            supervision = MechanismSupervision.restore(packet, receipt["supervision"])
            outcomes = supervision.record()["outcomes"]
            _require(
                all(t is None or t["available_at"] < config.fit_end for t in outcomes),
                "La supervisión contiene resultados posteriores al ajuste",
            )
            metrics = supervision.metrics(device=device)
            record["collections"].append(
                dict(
                    generation=entry["generation"],
                    packet_reference=entry["packet"],
                    receipt_reference=entry["receipt"],
                    collection_sha256=packet.sha256,
                    observation_sha256=body["observation"]["sha256"],
                )
            )
            _append_cohorts(observation, metrics, packet.sha256, config, used, cohort_keys, budgets)
        _require(source.snapshot() == before, "La sesión fuente cambió durante el ajuste")
        records.append(record)
    return _finish_cohorts(used, records, budgets, config)


def _append_cohorts(observation, metrics, packet_sha, config, used, cohort_keys, budgets):
    for group, metric in zip(observation["groups"], metrics, strict=True):
        budgets[1] += 1
        _require(budgets[1] <= config.max_cohorts, "Las fuentes superan las cohortes declaradas")
        market, at = group["market"], observation["cutoff"]
        _require(
            market in dict(config.market_weights) and (market, at) not in cohort_keys,
            "El mercado no tiene peso declarado o repite su decisión",
        )
        cohort_keys.add((market, at))
        if metric["mature_rows"]:
            used.append(
                dict(
                    features=group["values"],
                    market=market,
                    decision_at=at,
                    point=metric["point_gains"],
                    interval=metric["interval_gains"],
                    mature_rows=metric["mature_rows"],
                    excluded_rows=metric["excluded_rows"],
                    collection_sha256=packet_sha,
                )
            )


def _window_cohorts(sources, *, config, device):
    require_learning_allowed("lectura de señales por ventanas para la política de mecanismos")
    records, used, cohort_keys, budgets = [], [], set(), [0, 0]
    for collections, record in mechanism_fit_window.iter_fit_windows(sources, config=config):
        budgets[0] += record["source_bytes"]
        for packet, supervision, _ in collections:
            observation = packet.payload()["observation"]["observation"]
            _append_cohorts(
                observation,
                supervision.metrics(device=device),
                packet.sha256,
                config,
                used,
                cohort_keys,
                budgets,
            )
        records.append(record)
    rows, weights, provenance = _finish_cohorts(used, records, budgets, config)
    return rows, weights, dict(provenance, source_mode="native_windows")


def _finish_cohorts(used, records, budgets, config):
    market_weights = dict(config.market_weights)
    _require(len(used) >= config.min_cohorts, "No hay cohortes maduras suficientes para ajustar")
    present = {row["market"] for row in used}
    _require(present == set(market_weights), "Un mercado declarado no tiene cohortes maduras")
    counts = {market: sum(row["market"] == market for row in used) for market in present}
    used.sort(key=lambda row: (row["decision_at"], row["market"]))
    weights = [market_weights[row["market"]] / counts[row["market"]] for row in used]
    return (
        used,
        weights,
        dict(
            sources=records,
            source_bytes=budgets[0],
            observed_cohorts=budgets[1],
            no_label_cohorts=budgets[1] - len(used),
            cohorts_per_market=counts,
        ),
    )


def fit_policy(sources, *, config, device="cuda:0"):
    """Ajustar objetivos firmados en CUDA, sin convertir ausencias en observaciones.

    La modalidad identifica fases cerradas o prefijos nativos explícitos. En ambos casos
    la recepción debe preceder al final de ajuste. La selección del padre sigue siendo
    una declaración del protocolo que debe respaldar el estudio.
    """
    require_learning_allowed("ajuste de la política del catálogo de mecanismos")
    _require(type(config) is MechanismFitConfig, "Falta el protocolo identificado del ajuste")
    config.__post_init__()
    _require(device == "cuda:0", "El ajuste requiere cuda:0 explícito, sin sustitución por CPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible para ajustar la política")
    torch.cuda.set_device("cuda:0")
    _require(not torch.is_autocast_enabled("cuda"), "El ajuste requiere FP64 sin autocast")
    rows, row_weights, provenance = _cohorts(sources, config=config, device=device)
    with torch.no_grad():
        x = torch.tensor([row["features"] for row in rows], dtype=torch.float64, device=device)
        weights = torch.tensor(row_weights, dtype=torch.float64, device=device)
        weights /= weights.sum()
        feature_magnitude = x.abs().amax(0).clamp_min(1.0)
        scaled = x / feature_magnitude
        center = (weights[:, None] * scaled).sum(0)
        spread = (weights[:, None] * (scaled - center).square()).sum(0).sqrt()
        lower, upper = x.amin(0), x.amax(0)
        constant = lower == upper
        mean = torch.where(constant, lower, center * feature_magnitude)
        scale = spread * feature_magnitude
        scale = torch.where(constant | (scale == 0), torch.ones_like(scale), scale)
        centered = x - mean
        check_finite(mean, "La media de ajuste")
        check_finite(scale, "La escala de ajuste")
        check_finite(centered, "El centrado usado también en inferencia")
        _require(
            bool(((lower <= mean) & (mean <= upper)).all().item()),
            "La media de ajuste queda fuera del soporte observado",
        )
        z = torch.cat((centered / scale, torch.ones_like(x[:, :1])), dim=1)
        check_finite(z, "El diseño normalizado de ajuste")
        objectives = [row["point"] + row["interval"] for row in rows]
        # El cero solo rellena almacenamiento. Ningún objetivo ausente entra en un sistema.
        masks = torch.tensor(
            [[v is not None for v in row] for row in objectives], dtype=torch.bool, device=device
        )
        targets = torch.tensor(
            [[0.0 if v is None else v for v in row] for row in objectives],
            dtype=torch.float64,
            device=device,
        )
        check_finite(targets, "Las ganancias maduras del ajuste")
        coefficients = torch.zeros(
            (len(FEATURES) + 1, len(POINT_ACTIONS) + len(ACTIONS)),
            dtype=torch.float64,
            device=device,
        )
        counts = masks.sum(0).cpu().tolist()
        conditions = [None] * coefficients.shape[1]
        stationarity = [None] * coefficients.shape[1]
        penalty = float(config.ridge) ** 0.5 * torch.eye(z.shape[1], dtype=z.dtype, device=device)
        for column, count in enumerate(counts):
            if column in (0, len(POINT_ACTIONS)) or count < config.min_cohorts:
                continue
            chosen = masks[:, column]
            w = weights[chosen]
            w = w / w.sum()
            design = torch.cat((w.sqrt()[:, None] * z[chosen], penalty))
            singular = torch.linalg.svdvals(design)
            condition = float((singular[0] / singular[-1]).item())
            _require(
                condition <= config.max_condition, "El sistema de ajuste está mal condicionado"
            )
            values = targets[chosen, column]
            target_magnitude = values.abs().amax().clamp_min(1.0)
            rhs = torch.cat((w.sqrt() * (values / target_magnitude), z.new_zeros(z.shape[1])))
            normalized = torch.linalg.lstsq(design, rhs, driver="gels").solution
            residual = design.T @ (design @ normalized - rhs)
            check_finite(residual, "El residuo de estacionariedad del ajuste")
            stationarity[column] = float(torch.linalg.vector_norm(residual).item())
            fitted = normalized * target_magnitude
            check_finite(fitted, "Los coeficientes ridge ajustados")
            coefficients[:, column] = fitted
            conditions[column] = condition
        action_mask = torch.tensor(
            [
                action == 0
                or (
                    counts[action // 3] >= config.min_cohorts
                    and (
                        config.interval_weight == 0
                        or counts[len(POINT_ACTIONS) + action] >= config.min_cohorts
                    )
                )
                for action in range(len(ACTIONS))
            ],
            dtype=torch.bool,
            device=device,
        )
        policy = MechanismPolicy(
            mean=mean,
            scale=scale,
            point_coefficients=coefficients[:, : len(POINT_ACTIONS)],
            interval_coefficients=coefficients[:, len(POINT_ACTIONS) :],
            support_min=lower,
            support_max=upper,
            costs=x.new_tensor(config.costs),
            interval_weight=config.interval_weight,
            min_gain=config.min_gain,
            action_mask=action_mask,
            catalog_id=config.catalog_id,
        )
    receipt = dict(
        schema_version=1,
        kind="native_cohort_mechanism_fit_receipt_v1",
        protocol=config.identity(),
        protocol_sha256=config.fingerprint(),
        provenance=provenance,
        cohorts=len(rows),
        decision_instants=len({row["decision_at"] for row in rows}),
        cohorts_used=[
            {k: v for k, v in row.items() if k not in ("features", "point", "interval")}
            for row in rows
        ],
        objective_counts=counts,
        augmented_system_conditions=conditions,
        normalized_stationarity_norms=stationarity,
        mathematical_rank="positive_ridge_full_column_rank_in_exact_arithmetic",
        numerical_rank="not_reported_by_cuda_gels_not_a_certificate",
        solve_dtype="torch.float64",
        device=device,
        torch=str(torch.__version__),
        policy_buffers_sha256=policy_signature(policy, strong=True)[1]["buffers_sha256"],
        policy_configuration_without_receipt=policy.get_extra_state(),
        implementation={
            "fit": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "contract": hashlib.sha256(
                Path(mechanism_fit_contract.__file__).read_bytes()
            ).hexdigest(),
        },
        selection_performed=False,
        predictive_improvement_measured=False,
    )
    if config.source_mode == "native_windows":
        receipt.update(schema_version=2, kind="native_window_cohort_mechanism_fit_receipt_v1")
        receipt["implementation"]["windows"] = hashlib.sha256(
            Path(mechanism_fit_window.__file__).read_bytes()
        ).hexdigest()
    reference = digest(receipt)
    linked = MechanismPolicy(
        **dict(policy.named_buffers()),
        interval_weight=config.interval_weight,
        min_gain=config.min_gain,
        fit_receipt_sha256=reference,
        catalog_id=config.catalog_id,
    )
    return linked, dict(receipt, receipt_sha256=reference)
