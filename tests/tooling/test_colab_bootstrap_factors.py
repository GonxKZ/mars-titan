"""El arranque de Colab restaura el factor de China y se niega a seguir si falta un factor."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "colab_bootstrap.sh"
INSIDE = ".local/state/mars-titan/history/csi300/prices.parquet"


def run(step, destination, *arguments):
    command = ["bash", str(SCRIPT), step, "--destino", str(destination), *arguments]
    return subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)


def view(destination, spy, china):
    """Una vista conjunta con un activo de cada mercado y sus dos factores declarados."""
    folder = destination / ".local/state/mars-titan/campaign/views/US+CN/fold-000"
    folder.mkdir(parents=True)
    factors = {
        market: dict(market=market, prices_path=str(path), prices_sha256=digest)
        for market, (path, digest) in dict(US=spy, CN=china).items()
    }
    assets = [dict(market="US", symbol="AAA"), dict(market="CN", symbol="000001.SZ")]
    manifest = dict(assets=assets, market_factors=factors, final_test_opened=False)
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def factor_zip(path, payload, declared=None):
    sums = f"{declared or hashlib.sha256(payload).hexdigest()}  {INSIDE}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(INSIDE, payload)
        archive.writestr("SHA256SUMS.txt", sums)
        archive.writestr("LEEME.txt", "no se restaura")


def test_a_missing_market_factor_stops_the_bootstrap_and_the_zip_restores_it(tmp_path):
    destination = tmp_path / "home"
    spy = destination / ".local/state/mars-titan/prepared/US/SPY/prices.parquet"
    spy.parent.mkdir(parents=True)
    spy.write_bytes(b"spy")
    payload = b"indice csi 300"
    china = destination / INSIDE
    view(
        destination,
        (spy, hashlib.sha256(b"spy").hexdigest()),
        (china, hashlib.sha256(payload).hexdigest()),
    )
    # Sin el factor de China el paso se detiene y lo nombra.
    missing = run("factores", destination)
    assert missing.returncode != 0 and "CN" in missing.stderr and "FALTA" in missing.stderr
    assert "factor de CN" in missing.stderr and "factor de US" not in missing.stderr
    # Un zip cuya huella no cuadra no se restaura.
    bad = tmp_path / "malo.zip"
    factor_zip(bad, payload, declared="0" * 64)
    refused = run("factor-cn", destination, "--factor-cn", str(bad))
    assert refused.returncode != 0 and not china.exists()
    # El zip bueno deja el fichero en su ruta y la comprobación pasa.
    good = tmp_path / "bueno.zip"
    factor_zip(good, payload)
    restored = run("factor-cn", destination, "--factor-cn", str(good))
    assert restored.returncode == 0, restored.stderr
    assert china.read_bytes() == payload and not (destination / "LEEME.txt").exists()
    assert run("factores", destination).returncode == 0
    # Repetirlo no cambia nada, y un fichero existente con otros bytes detiene el paso.
    assert run("factor-cn", destination, "--factor-cn", str(good)).returncode == 0
    china.write_bytes(b"otro contenido")
    clash = run("factor-cn", destination, "--factor-cn", str(good))
    assert clash.returncode != 0 and china.read_bytes() == b"otro contenido"
    # Con el fichero cambiado, la huella declarada ya no cuadra.
    changed = run("factores", destination)
    assert changed.returncode != 0 and "CN" in changed.stderr


def test_a_zip_with_a_path_outside_the_destination_is_refused(tmp_path):
    destination = tmp_path / "home"
    destination.mkdir()
    archive = tmp_path / "fuera.zip"
    payload = b"x"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(".local/../../fuera.parquet", payload)
        name = ".local/../../fuera.parquet"
        bundle.writestr("SHA256SUMS.txt", f"{hashlib.sha256(payload).hexdigest()}  {name}\n")
    refused = run("factor-cn", destination, "--factor-cn", str(archive))
    assert refused.returncode != 0 and not (tmp_path / "fuera.parquet").exists()
