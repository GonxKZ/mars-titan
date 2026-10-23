"""Decisión de cohorte y recibo reproducible del catálogo cerrado."""

import hashlib
import inspect
from dataclasses import asdict

import torch

from .config import canonical
from .financial import _tensor_digest
from .mechanism_catalog import MechanismAction, MechanismAvailability
from .mechanism_observation import MechanismObservation
from .mechanism_policy import ACTIONS, MechanismPolicy


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def policy_signature(policy, *, strong):
    """Comprobar componentes congelados sin admitir hooks ni métodos sustituidos."""
    if type(policy) is not MechanismPolicy or policy.training:
        raise ValueError("El selector debe conservar su tipo y modo eval")
    if len(list(policy.modules())) != 1 or list(policy.named_parameters()):
        raise ValueError("El selector solo admite sus buffers congelados")
    for name, value in vars(policy).items():
        if (name.endswith("_hooks") and value) or (
            callable(value) and hasattr(MechanismPolicy, name)
        ):
            raise ValueError("El selector no admite hooks ni métodos sustituidos")
    methods = tuple(
        (name, id(method), id(method.__code__))
        for name, method in inspect.getmembers(MechanismPolicy, inspect.isfunction)
    )
    buffers = []
    for name, value in policy.named_buffers():
        if value.requires_grad or value.grad is not None or value.grad_fn is not None:
            raise ValueError("Los buffers del selector deben estar congelados")
        buffers.append(
            (
                name,
                id(value),
                value._version,
                value.dtype,
                value.device,
                value.shape,
                value.stride(),
            )
        )
    identity = dict(
        configuration=policy.get_extra_state(),
        coefficient_origin=(
            "supplied_not_verified_as_fitted"
            if policy.fit_receipt_sha256 is None
            else "supplied_with_declared_fit_receipt_not_independently_verified"
        ),
        dtype=str(policy.mean.dtype),
        device=str(policy.mean.device),
    )
    if strong:
        policy.validate()
        identity["buffers_sha256"] = digest(
            {name: _tensor_digest(value) for name, value in policy.named_buffers()}
        )
    return (methods, tuple(buffers), canonical(policy.get_extra_state())), identity


def select(consumer, observation, *, state, flows, cutoff, context_id, snapshot, availability):
    """Elegir una vez por mercado, antes de MAC y de cualquier feedback actual."""
    if type(observation) is not MechanismObservation:
        raise ValueError("El selector necesita una observación de cohorte sellada")
    body = observation.record()["observation"]
    expected = dict(
        consumer_id=consumer.model_id,
        parameter_id=consumer.predictor._parameter_id,
        input_contract_id=consumer.predictor.config.inputs.fingerprint(),
        input_policy=consumer.predictor.config.inputs.input_policy,
        context_id=context_id,
        cutoff=cutoff,
        previous_cutoff=state.cutoff,
        flow_ids_sha256=digest(flows),
        snapshot_sha256=snapshot._digest if snapshot is not None else None,
        bank_entries=snapshot.count if snapshot is not None else 0,
    )
    groups = [g["market"] for g in body["groups"]]
    if any(body[key] != value for key, value in expected.items()) or groups != list(availability):
        raise ValueError("La observación no corresponde al consumidor, corte o snapshot anterior")
    for group in body["groups"]:
        if group["rows"] != sum(flow.startswith(group["market"] + "/") for flow in flows):
            raise ValueError("La observación no conserva las filas de cada mercado")
    actions = consumer.catalog.actions()
    resolutions = [
        [consumer.catalog.resolve(a, availability[g["market"]], rows=g["rows"]) for a in actions]
        for g in body["groups"]
    ]
    policy = consumer.policy
    features = observation.tensor(device=policy.mean.device, dtype=policy.mean.dtype)
    allowed = torch.tensor(
        [[not result.reasons for result in row] for row in resolutions],
        dtype=torch.bool,
        device=features.device,
    )
    choice = policy(features, allowed)
    # Una transferencia por matriz mantiene acotado el coste, sin .item() por acción.
    arrays = {
        key: value.detach().cpu().tolist()
        for key, value in dict(
            features=features,
            standardized=choice.standardized,
            action_id=choice.action_ids,
            scores=choice.scores,
            point_gains=choice.point_gains,
            interval_gains=choice.interval_gains,
            supported=choice.supported,
            available=choice.available,
        ).items()
    }
    records, plans = [], []
    for i, group in enumerate(groups):
        action_id = arrays["action_id"][i]
        plans.append((group, resolutions[i][action_id]))
        records.append(
            dict(
                market=group,
                action=ACTIONS[action_id],
                availability=asdict(availability[group]),
                **{key: value[i] for key, value in arrays.items()},
            )
        )
    trace = dict(
        schema_version=1,
        kind="pre_mac_mechanism_selection_v1",
        policy_sha256=digest(consumer._policy_identity),
        dtype=str(policy.mean.dtype),
        observation=observation.record(),
        groups=records,
    )
    return tuple(plans), canonical(trace)


def _calibration_failed(work, actions):
    names = (
        "online_attempts",
        "online_failures",
        "online_created_proposals",
        "online_selected_proposals",
        "online_discarded_proposals",
    )
    if any(type(work.get(name)) is not int or work[name] < 0 for name in names):
        raise ValueError("La traza no conserva el trabajo de calibración")
    attempts, failures, created, selected, discarded = (work[name] for name in names)
    online = sum(action.calibration == "online" for action in actions)
    if failures == 0:
        valid = attempts == created == selected == online and discarded == 0
    else:
        valid = (
            failures == 1
            and 1 <= attempts <= online
            and created <= attempts
            and selected == 0
            and discarded == created
        )
    if not valid:
        raise ValueError("La selección no concilia las propuestas publicadas o descartadas")
    return failures == 1


def _execution_matches(row, resolved, failed):
    action = resolved.applied
    fallback = failed and action.calibration == "online"
    common = (
        row["requested"] == action.key
        and row["context"] == action.context
        and row["executed_route"] == action.route
        and row["reader_steps"] == resolved.reader_steps
        and row["reader_searches"] == resolved.reader_searches
    )
    if fallback:
        return (
            common
            and row["emitted_route"] == "k1"
            and row["calibration"] == "none"
            and tuple(row["reasons"]) == ("pending_calibration_bytes",)
            and row["point"] == row["k1_point"]
            and row["raw_quantiles"] == row["k1_quantiles"]
            and row["discarded_reader_steps"] == max(0, resolved.reader_steps - 1)
        )
    return (
        common
        and row["emitted_route"] == action.route
        and row["calibration"] == action.calibration
        and not row["reasons"]
        and row["discarded_reader_steps"] == 0
    )


def check_trace(consumer, trace, *, rows, cutoff, context_id, snapshot_sha256, work):
    """Conciliar el recibo con las filas emitidas y recalcular utilidades congeladas.

    La verificación no autentica un host capaz de modificar el registro y el código.
    La admisión física y el estado anterior se comprueban al producir la propuesta.
    """
    if type(trace) is not dict or set(trace) != {
        "schema_version",
        "kind",
        "policy_sha256",
        "dtype",
        "observation",
        "groups",
    }:
        raise ValueError("La selección no conserva su esquema completo")
    packet = MechanismObservation.restore(trace["observation"])
    body = packet.record()["observation"]
    fields = ("flow_id", "sample_id", "prediction_at", "input_available_at", "input_sha256")
    expected = dict(
        consumer_id=consumer.model_id,
        parameter_id=consumer.predictor._parameter_id,
        input_contract_id=consumer.predictor.config.inputs.fingerprint(),
        input_policy=consumer.predictor.config.inputs.input_policy,
        context_id=context_id,
        cutoff=cutoff,
        snapshot_sha256=snapshot_sha256,
        flow_ids_sha256=digest([row["flow_id"] for row in rows]),
        rows_sha256=digest([{name: row[name] for name in fields} for row in rows]),
    )
    if (
        type(trace["schema_version"]) is not int
        or trace["schema_version"] != 1
        or trace["kind"] != "pre_mac_mechanism_selection_v1"
        or trace["policy_sha256"] != digest(consumer._policy_identity)
        or trace["dtype"] != str(consumer.policy.mean.dtype)
        or any(body[key] != value for key, value in expected.items())
        or type(trace["groups"]) is not list
        or len(trace["groups"]) != len(body["groups"])
    ):
        raise ValueError("La selección contradice sus observaciones, política o identidad")
    policy = consumer.policy
    features = packet.tensor(device=policy.mean.device, dtype=policy.mean.dtype)
    masks = []
    for group, source in zip(trace["groups"], body["groups"], strict=True):
        try:
            available = MechanismAvailability(**group["availability"])
        except (TypeError, KeyError) as error:
            raise ValueError("Falta la disponibilidad de la selección") from error
        masks.append(
            [
                not consumer.catalog.resolve(a, available, rows=source["rows"]).reasons
                for a in consumer.catalog.actions()
            ]
        )
    choice = policy(features, torch.tensor(masks, dtype=torch.bool, device=features.device))
    actual = dict(
        features=features,
        standardized=choice.standardized,
        action_id=choice.action_ids,
        scores=choice.scores,
        point_gains=choice.point_gains,
        interval_gains=choice.interval_gains,
        supported=choice.supported,
        available=choice.available,
    )
    actual = {key: value.detach().cpu().tolist() for key, value in actual.items()}
    actions = [MechanismAction(*ACTIONS[index].split("/")) for index in actual["action_id"]]
    failed = _calibration_failed(work, actions)
    for i, (group, source) in enumerate(zip(trace["groups"], body["groups"], strict=True)):
        action = actions[i]
        resolved = consumer.catalog.resolve(
            action, MechanismAvailability(**group["availability"]), rows=source["rows"]
        )
        subset = [row for row in rows if row["flow_id"].startswith(source["market"] + "/")]
        if (
            set(group) != {"market", "action", "availability", *actual}
            or group["market"] != source["market"]
            or len(subset) != source["rows"]
            or canonical({key: group[key] for key in actual})
            != canonical({key: value[i] for key, value in actual.items()})
            or group["action"] != action.key
            or any(not _execution_matches(row, resolved, failed) for row in subset)
        ):
            raise ValueError("La elección, admisibilidad o utilidad no reproduce la emisión")
