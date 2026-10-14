"""Compara bit a bit dos capturas de parity_capture.py.

Uso: parity_compare.py BASE.pt NUEVA.pt. Exige la misma auditoría (predicciones, etiquetas,
pérdidas y demás entradas, con igualdad exacta de flotantes) y los mismos gradientes de cada
llamada al optimizador sustituto, con su tipo y su nombre.
"""

import json
import math
import sys

import torch


def same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def without_code(identity):
    return {key: value for key, value in identity.items() if key != "implementation"}


base, new = (torch.load(path, weights_only=False) for path in sys.argv[1:3])
assert (base["device"], base["dtype"]) == (new["device"], new["dtype"])
report = {}
for key in base["captures"]:
    a, b = base["captures"][key], new["captures"][key]
    report[key] = dict(
        audit_entries=len(a["audit"]),
        predictions=sum(entry[0] == "prediction" for entry in a["audit"]),
        optimizer_calls=a["calls"],
        same_audit=same(a["audit"], b["audit"]),
        same_gradients=same(a["grads"], b["grads"]),
        same_calls=a["calls"] == b["calls"],
        same_metrics=same(a.get("metrics"), b.get("metrics")),
        closed_passes=len(a["states"]),
        same_states=same(a["states"], b["states"]),
        # La identidad incluye el SHA-256 de los módulos, que cambia con el código medido.
        same_identity=same(without_code(a["identity"]), without_code(b["identity"])),
        changed_modules=sorted(
            name
            for name, value in a["identity"].get("implementation", {}).items()
            if b["identity"].get("implementation", {}).get(name) != value
        ),
        same_report=same(a.get("report"), b.get("report")),
    )
report = dict(
    device=base["device"],
    dtype=base["dtype"],
    same_keys=base["captures"].keys() == new["captures"].keys(),
    captures=report,
)
print(json.dumps(report, indent=1))
ok = report["same_keys"] and all(
    all(value for name, value in item.items() if name.startswith("same_"))
    for item in report["captures"].values()
)
sys.exit(0 if ok else 1)
