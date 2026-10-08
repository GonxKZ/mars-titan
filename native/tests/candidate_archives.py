"""Crear ZIP adversos acotados y comprobar su rechazo antes de cargar tensores."""

import hashlib
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import torch

cli, checker, directory = sys.argv[1:]
root = Path(directory)
root.mkdir(parents=True, exist_ok=True)
valid = root / "valid.pt"
valid.unlink(missing_ok=True)
subprocess.run([cli, "cpu", str(valid)], check=True)
subprocess.run([checker, "valid", str(valid)], check=True)
block = bytes(1024**2)


def rewrite(name, extensions, duplicate=False):
    target = root / name
    with (
        zipfile.ZipFile(valid) as source,
        zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as destination,
    ):
        for info in source.infolist():
            original = source.read(info.filename)
            padding = extensions.get(info.filename, 0)
            if not padding:
                destination.writestr(info.filename, original)
                continue
            with destination.open(info.filename, "w") as stream:
                stream.write(original)
                while padding:
                    size = min(len(block), padding)
                    stream.write(block[:size])
                    padding -= size
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                first = source.namelist()[0]
                name = first.upper() if duplicate == "case" else first
                destination.writestr(name, source.read(first))
    return target


with zipfile.ZipFile(valid) as source:
    names = source.namelist()
    records = [name for name in names if "/data/" in name]
    version = next(name for name in names if name.endswith("/version"))
    identity = next(name for name in names if name.endswith("/.data/serialization_id"))

cases = [
    ("storage.pt", {records[0]: 160 * 1024**2}),
    ("aggregate.pt", {records[0]: 70 * 1024**2, records[1]: 70 * 1024**2}),
    ("identity.pt", {identity: 160 * 1024**2}),
    ("version.pt", {version: 160 * 1024**2}),
]
for filename, changes in cases:
    archive = rewrite(filename, changes)
    subprocess.run([checker, "descomprimido", str(archive)], check=True)
repeated = rewrite("duplicate.pt", {}, duplicate=True)
subprocess.run([checker, "duplicado", str(repeated)], check=True)
case_collision = rewrite("case-duplicate.pt", {}, duplicate="case")
subprocess.run([checker, "duplicado", str(case_collision)], check=True)
many = rewrite("many-records.pt", {})
with zipfile.ZipFile(many, "a") as destination:
    for index in range(1025):
        destination.writestr(f"archive/extra-{index}", b"")
subprocess.run([checker, "registros ZIP", str(many)], check=True)
truncated = root / "truncated.pt"
truncated.write_bytes(valid.read_bytes()[:-32])
subprocess.run([checker, "ZIP", str(truncated)], check=True)

# Una proyección cargada no debe compartir almacenamiento con un parámetro público.
module = torch.jit.load(str(valid), map_location="cpu")
module.key_projection = module.network.initial_weight.t()
digest = hashlib.sha256(module.key_projection.detach().contiguous().numpy().tobytes()).hexdigest()
module.representation_id = module.representation_id.rsplit(":", 1)[0] + ":" + digest
aliased = root / "aliased-projection.pt"
torch.jit.save(module, str(aliased))
subprocess.run([checker, "alias", str(aliased)], check=True)
