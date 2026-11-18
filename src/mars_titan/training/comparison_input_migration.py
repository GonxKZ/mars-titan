"""Trasladar una preparación de paso cero con los mismos pesos, optimizador y RNG."""

import copy
import hashlib
import re
import struct
import subprocess
from functools import lru_cache
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.titans.config import canonical

GPU_RELEASE = "eb01b883e8e6ec912e2cda85f738fd2525584d19"
WARMUP_RELEASE = "e23a44034c8584682397d3d5f7ac76dbc7bd8e12"
PACKED_RELEASE = "f52cc3c332ba358e7afc1772e2038f05505c2285"
_CHANGED = frozenset(
    {
        "mars_titan.training.financial_run",
        "mars_titan.training.mars_titan_run",
        "mars_titan.training.titans_walk_forward",
        "mars_titan.training.window_indices",
        "mars_titan.memory.write_scores",
        "mars_titan.models.titans.state",
    }
)
_NEW_INPUT = ("mars_titan.memory.event_archive", "mars_titan.memory.training_event_cache")
_MODULE = re.compile(r"mars_titan(?:\.[A-Za-z_][A-Za-z_0-9]*)+")


def _require(value, message):
    if not value:
        raise ValueError(message)


def run_id(identity):
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


def _path(root, module):
    if module == "data/catalogs/macro-indicators.csv":
        return Path(root) / module
    _require(isinstance(module, str) and _MODULE.fullmatch(module), "Nombre de módulo no válido")
    file = Path(root) / "src" / (module.replace(".", "/") + ".py")
    return file if file.is_file() else Path(root) / "src" / module.replace(".", "/") / "__init__.py"


@lru_cache(maxsize=64)
def git_digest(root, commit, module):
    """Huella de los bytes del módulo en el commit indicado."""
    root = Path(root)
    relative = _path(root, module).relative_to(root).as_posix()
    proc = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative}"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return hashlib.sha256(proc.stdout).hexdigest() if proc.returncode == 0 else None


def equivalent(root, module, previous, current):
    return (
        module in _CHANGED
        and isinstance(previous, str)
        and re.fullmatch(r"[0-9a-f]{64}", previous) is not None
        and current is not None
        and current == git_digest(root, PACKED_RELEASE, module)
        and any(
            previous == git_digest(root, version, module)
            for version in (WARMUP_RELEASE, GPU_RELEASE)
        )
    )


def _translate(root, modules):
    result = {}
    for name, old in modules.items():
        file = _path(root, name)
        _require(file.is_file(), f"Falta el módulo {name}")
        new = sha256(file)
        _require(
            old == new or equivalent(root, name, old, new),
            f"El código de {name} no está cualificado",
        )
        result[name] = new
    return result


def same_state(a, b):
    """Comparar también los bits de los tensores, incluidos los ceros con signo."""
    import torch

    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return (
            a.dtype == b.dtype
            and a.shape == b.shape
            and a.stride() == b.stride()
            and torch.equal(
                a.contiguous().reshape(-1).view(torch.uint8),
                b.contiguous().reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_state(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float):
        return struct.pack("!d", a) == struct.pack("!d", b)
    return a == b


def _read(path):
    return read_manifest(path, 16 * 1024**2)[0]


def transfer_initial(folder, dest, root, *, host_flags, archived=True):
    from .checkpoints import load_training_state, save_training_state

    outer = _read(folder / "run.json")
    fit = _read(folder / "fit/run.json")
    cursor = {"epoch": 0, "phase": "validation"}
    _require(
        outer.get("kind") == "titans_walk_forward_window"
        and outer.get("status") == "paused"
        and outer.get("final_test_opened") is False
        and not outer.get("predictions")
        and fit.get("kind") == "titans_financial_chronological_run"
        and fit.get("status") == "paused"
        and fit.get("final_test_opened") is False
        and fit.get("global_step") == 0
        and fit.get("cursor") == cursor
        and fit.get("history") == []
        and fit.get("selection") is None,
        "Solo se traslada una preparación de paso cero",
    )
    identity = fit["identity"]
    _require(
        fit["run_id"] == run_id(identity)
        and outer["run_id"] == run_id(outer["identity"])
        and outer["identity"]["request"] == outer["request"]
        and outer["identity"]["fit_run_id"] == fit["run_id"],
        "La identidad del punto de partida ha cambiado",
    )
    original = folder / "fit/checkpoints"
    index = _read(original / "latest.json")
    _require(
        len(index["latest"]) == 1
        and index["latest"][0]["global_step"] == 0
        and index["best"] is None
        and not index["pinned"],
        "El paso cero contiene otra selección o confirmación",
    )
    ck = index["latest"][0]
    old_sha = ck["sha256"]
    _require(
        fit["recovery_checkpoint"] == dict(path="checkpoints/" + ck["name"], sha256=old_sha),
        "El informe no identifica el checkpoint inicial",
    )
    state = load_training_state(original, expected_identity=identity, expected_sha256=old_sha)
    _require(
        state["global_step"] == 0
        and state["cursor"] == cursor
        and state["run"] is None
        and state["selection"] is None
        and state["history"] == []
        and state["train_metrics"] is None
        and state["optimizer"]["state"] == {}
        and "rng" in state
        and "model" in state,
        "El paso cero contiene trabajo pendiente o actualizaciones",
    )
    new_identity = copy.deepcopy(identity)
    modules = _translate(root, identity["implementation"])
    extra = (
        *(_NEW_INPUT if archived else ()),
        *(("mars_titan.models.titans.host_finite",) if host_flags else ()),
    )
    for name in extra:
        actual = sha256(_path(root, name))
        _require(
            actual == git_digest(root, PACKED_RELEASE, name),
            f"El módulo adicional {name} no coincide con la versión",
        )
        modules[name] = actual
    new_identity["implementation"] = modules
    request = dict(outer["request"], code=_translate(root, outer["request"]["code"]))
    dest.mkdir(parents=True, exist_ok=False)
    target = dest / "fit/checkpoints"
    new_file = save_training_state(target, state, identity=new_identity)
    new_sha = sha256(new_file)
    confirmed = load_training_state(target, expected_identity=new_identity, expected_sha256=new_sha)
    _require(same_state(state, confirmed), "El traslado ha cambiado el contenido del checkpoint")
    receipt = dict(
        global_step=0,
        archived_inputs=archived,
        host_finite_flags=host_flags,
        source_fit_run_id=fit["run_id"],
        source_window_run_id=outer["run_id"],
        source_checkpoint_sha256=old_sha,
        checkpoint_sha256=new_sha,
    )
    new_fit = dict(
        fit,
        identity=new_identity,
        run_id=run_id(new_identity),
        input_migration=receipt,
        recovery_checkpoint=dict(path="checkpoints/" + new_file.name, sha256=new_sha),
    )
    window = dict(outer["identity"], request=request, fit_run_id=new_fit["run_id"])
    atomic_json(dest / "fit/run.json", new_fit)
    atomic_json(
        dest / "run.json",
        dict(
            outer, identity=window, request=request, run_id=run_id(window), input_migration=receipt
        ),
    )
    return receipt
