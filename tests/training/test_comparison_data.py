"""Paquete de datos de la campaña de comparación con archivos pequeños de prueba (#836)."""

import hashlib
import json
import shutil
import tarfile

import pytest

from mars_titan.training import comparison_data as data
from mars_titan.training.comparison_campaign import Layout

ORIGINS = [dict(year=2019, fold="fold-014"), dict(year=2020, fold="fold-015")]
SYMBOLS = {2019: ["AA", "BB"], 2020: ["BB", "CC"]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, text, *, repeat=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode() * repeat)
    return path


def write_json(path, document):
    return write(path, json.dumps(document), repeat=1)


@pytest.fixture
def campaign(tmp_path):
    """Dos orígenes con cohortes solapadas, sus vistas, el factor de mercado y el enlace."""
    home = tmp_path / "home"
    state = home / "state"
    prepared, samples = state / "edition/prepared", state / "edition/samples"
    place = Layout(state / "campaign", state / "views/US")
    factor = write(prepared / "US/SPY/prices.parquet", "factor")
    parent = write_json(state / "views/parents/US.json", {})
    for symbol in ("AA", "BB", "CC", "ZZ"):
        write(prepared / f"US/{symbol}/prices.parquet", f"prices-{symbol}")
        write(prepared / f"US/{symbol}/news/2019.parquet", f"news-{symbol}")
        write(samples / f"US/{symbol}/samples.parquet", f"samples-{symbol}")
        write(samples / f"US/{symbol}/manifest.json", f"receipt-{symbol}")
        write(samples / f"US/{symbol}/.asset.lock", "")
    for origin in ORIGINS:
        labels = place.source(origin["fold"]).parent / "labels"
        assets = []
        for symbol in SYMBOLS[origin["year"]]:
            path = write(
                labels / f"US/{symbol}/labels.parquet", f"labels-{origin['year']}-{symbol}"
            )
            assets.append(
                dict(
                    market="US",
                    symbol=symbol,
                    prices_sha256=digest(prepared / f"US/{symbol}/prices.parquet"),
                    samples_sha256=digest(samples / f"US/{symbol}/samples.parquet"),
                    labels_sha256=digest(path),
                )
            )
        roots = dict(prepared=str(prepared), samples=str(samples), labels=str(labels))
        view = dict(
            markets=["US"],
            market_factors=dict(
                US=dict(prices_path=str(factor), prices_sha256=digest(factor)),
                CN=dict(prices_path=str(home / "missing.parquet"), prices_sha256="0" * 64),
            ),
            temporal_view=dict(parent_manifest=str(parent)),
            parent_preparation=dict(path=str(home / "not-on-this-machine.json")),
        )
        write_json(place.source(origin["fold"]), view)
        write_json(place.cohort(origin["year"]) / "manifest.json", dict(roots=roots, assets=assets))
    native = write(home / "native/build/_episodic_native.so", "binding")
    write(home / "native/build/libsimulation.so", "library")
    write(home / "native/build/notes.txt", "not a library")
    return dict(home=home, place=place, declaration=dict(origins=ORIGINS), native=native)


def test_campaign_files_are_the_cohort_assets_their_views_and_the_market_factor(campaign):
    home, place = campaign["home"], campaign["place"]
    files = data.campaign_files(campaign["declaration"], place)
    names = sorted(path.relative_to(home).as_posix() for path in files)
    assert names == sorted(
        [
            "state/edition/prepared/US/SPY/prices.parquet",
            "state/views/parents/US.json",
            *(f"state/views/US/{origin['fold']}/manifest.json" for origin in ORIGINS),
            *(
                f"state/views/US/{origin['fold']}/labels/US/{symbol}/labels.parquet"
                for origin in ORIGINS
                for symbol in SYMBOLS[origin["year"]]
            ),
            *(
                f"state/edition/{folder}/US/{symbol}/{name}"
                for symbol in ("AA", "BB", "CC")
                for folder, name in (
                    ("prepared", "prices.parquet"),
                    ("prepared", "news/2019.parquet"),
                    ("samples", "samples.parquet"),
                    ("samples", "manifest.json"),
                )
            ),
        ]
    )
    # Ningún activo fuera de las cohortes, ningún cerrojo y nada del estado de la campaña.
    assert not any("ZZ" in name or ".asset.lock" in name or "campaign" in name for name in names)
    prices = home / "state/edition/prepared/US/BB/prices.parquet"
    assert files[prices] == digest(prices)
    assert files[home / "state/edition/prepared/US/BB/news/2019.parquet"] is None
    assert list(files) == sorted(files)
    with_native = data.campaign_files(campaign["declaration"], place, native=campaign["native"])
    extra = sorted(path.name for path in set(with_native) - set(files))
    assert extra == ["_episodic_native.so", "libsimulation.so"]


def test_campaign_files_need_prepared_cohorts_and_every_declared_file(campaign):
    place, declaration = campaign["place"], campaign["declaration"]
    (campaign["home"] / "state/edition/samples/US/CC/samples.parquet").unlink()
    with pytest.raises(ValueError, match="Falta el archivo"):
        data.campaign_files(declaration, place)
    shutil.rmtree(place.cohort(2020))
    with pytest.raises(ValueError, match="Falta la cohorte de 2020"):
        data.campaign_files(declaration, place)


def exported(campaign, tmp_path, **options):
    files = data.campaign_files(
        campaign["declaration"], campaign["place"], native=campaign["native"]
    )
    destination = tmp_path / "disk"
    destination.mkdir()
    manifest = data.export_bundle(files, destination, home=campaign["home"], **options)
    return files, destination, manifest


@pytest.mark.parametrize("part_bytes", [None, 1024**2])
def test_bundle_restores_every_file_bit_for_bit_and_checks_its_digest(
    campaign, tmp_path, part_bytes
):
    home = campaign["home"]
    files, destination, manifest = exported(
        campaign, tmp_path, part_bytes=part_bytes, identity=dict(declaration="d" * 64)
    )
    assert manifest == json.loads((destination / data.MANIFEST).read_text())
    assert manifest["home"] == str(home) and manifest["final_test_opened"] is False
    assert [record["path"] for record in manifest["files"]] == [
        path.relative_to(home).as_posix() for path in files
    ]
    assert all(
        record["sha256"] == digest(home / record["path"])
        and record["bytes"] == (home / record["path"]).stat().st_size
        for record in manifest["files"]
    )
    assert manifest["total_bytes"] == sum(path.stat().st_size for path in files)
    assert sorted(path.name for path in destination.glob("*.tar")) == manifest["parts"]
    assert len(manifest["parts"]) == 1
    originals = {path: path.read_bytes() for path in files}
    # El otro equipo empieza sin datos: solo conserva un archivo idéntico ya copiado.
    kept = home / "state/edition/prepared/US/SPY/prices.parquet"
    for path in files:
        if path != kept:
            path.unlink()
    result = data.import_bundle(destination, home=home)
    assert result == dict(
        files=len(files),
        restored=len(files) - 1,
        kept=1,
        total_bytes=manifest["total_bytes"],
        identity=dict(declaration="d" * 64),
    )
    assert {path: path.read_bytes() for path in files} == originals
    assert not list(home.rglob("*.importing"))
    # Repetir la importación no cambia nada.
    assert data.import_bundle(destination, home=home)["kept"] == len(files)


def test_bundle_splits_into_parts_no_larger_than_requested(campaign, tmp_path):
    home = campaign["home"]
    big = home / "state/edition/samples/US/BB/samples.parquet"
    big.write_bytes(b"s" * (700 * 1024))
    (home / "state/edition/samples/US/CC/samples.parquet").write_bytes(b"t" * (700 * 1024))
    for origin in ORIGINS:
        path = campaign["place"].cohort(origin["year"]) / "manifest.json"
        cohort = json.loads(path.read_text())
        for asset in cohort["assets"]:
            asset["samples_sha256"] = digest(
                home / f"state/edition/samples/US/{asset['symbol']}/samples.parquet"
            )
        path.write_text(json.dumps(cohort))
    files, destination, manifest = exported(campaign, tmp_path, part_bytes=1024**2)
    assert len(manifest["parts"]) == 2
    for part in manifest["parts"]:
        inside = sum(r["bytes"] for r in manifest["files"] if r["part"] == part)
        assert inside <= 1024**2
        assert (destination / part).stat().st_size <= 1024**2
    for path in files:
        path.unlink()
    assert data.import_bundle(destination, home=home)["restored"] == len(files)
    assert big.read_bytes() == b"s" * (700 * 1024)


def test_export_refuses_a_changed_file_a_used_destination_and_files_outside_home(
    campaign, tmp_path
):
    home, place = campaign["home"], campaign["place"]
    files = data.campaign_files(campaign["declaration"], place)
    disk = tmp_path / "disk"
    with pytest.raises(ValueError, match="carpeta existente"):
        data.export_bundle(files, disk, home=home)
    disk.mkdir()
    with pytest.raises(ValueError, match="fuera de la carpeta personal"):
        data.export_bundle(files, disk, home=home / "state/edition")
    with pytest.raises(ValueError, match="al menos 1 MiB"):
        data.export_bundle(files, disk, home=home, part_bytes=1024)
    with pytest.raises(ValueError, match="No hay archivos"):
        data.export_bundle({}, disk, home=home)
    (home / "state/edition/samples/US/AA/samples.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="no conserva la huella"):
        data.export_bundle(files, disk, home=home)
    # El intento fallido dejó un tar sin manifiesto: el destino ya no está vacío.
    assert list(disk.glob("*.tar")) and not (disk / data.MANIFEST).exists()
    with pytest.raises(ValueError, match="ya contiene un paquete"):
        data.export_bundle(files, disk, home=home)


def test_import_refuses_another_home_damage_and_different_existing_files(campaign, tmp_path):
    home = campaign["home"]
    files, destination, manifest = exported(campaign, tmp_path)
    with pytest.raises(ValueError, match="misma carpeta personal"):
        data.import_bundle(destination, home=tmp_path / "elsewhere")
    assert not (tmp_path / "elsewhere").exists()
    victim = home / "state/edition/samples/US/AA/samples.parquet"
    # La misma longitud obliga a comprobar el contenido, no solo el tamaño.
    local_edits = b"x" * victim.stat().st_size
    victim.write_bytes(local_edits)
    with pytest.raises(ValueError, match="ya existe con otro contenido"):
        data.import_bundle(destination, home=home)
    assert victim.read_bytes() == local_edits
    # Un byte cambiado dentro del tar se detecta al escribir y no deja el archivo.
    victim.unlink()
    part = destination / manifest["parts"][0]
    content = bytearray(part.read_bytes())
    with tarfile.open(part) as archive:
        member = archive.getmember("state/edition/samples/US/AA/samples.parquet")
    content[member.offset_data] ^= 0xFF
    part.write_bytes(bytes(content))
    with pytest.raises(ValueError, match="llegó dañado"):
        data.import_bundle(destination, home=home)
    assert not victim.exists() and not list(home.rglob("*.importing"))


def test_import_refuses_undeclared_members_missing_parts_and_invalid_manifests(campaign, tmp_path):
    home = campaign["home"]
    files, destination, manifest = exported(campaign, tmp_path)
    path = destination / data.MANIFEST

    def rewritten(change):
        document = json.loads(json.dumps(manifest))
        change(document)
        path.write_text(json.dumps(document))

    for change, message in (
        (lambda d: d.update(kind="other"), "no contiene un paquete"),
        (lambda d: d["files"][0].update(path="../outside"), "archivo no válido"),
        (lambda d: d["files"][0].update(path="/etc/passwd"), "archivo no válido"),
        (lambda d: d["files"].append(dict(d["files"][0])), "archivo no válido"),
        (lambda d: d["files"][0].update(part="other.tar"), "archivo no válido"),
        (lambda d: d["files"].pop(), "no declara así"),
        (lambda d: d["files"][0].update(bytes=1), "no declara así"),
    ):
        rewritten(change)
        with pytest.raises(ValueError, match=message):
            data.import_bundle(destination, home=home)
    rewritten(lambda d: None)
    (destination / manifest["parts"][0]).rename(destination / "moved.tar")
    with pytest.raises(ValueError, match="Falta la parte"):
        data.import_bundle(destination, home=home)
    (destination / "moved.tar").rename(destination / manifest["parts"][0])
    # Un paquete al que le falta un archivo declarado no se da por restaurado.
    rewritten(
        lambda d: d["files"].append(
            dict(path="state/extra.bin", bytes=1, sha256="0" * 64, part=d["parts"][0])
        )
    )
    with pytest.raises(ValueError, match="no trae 1 archivos"):
        data.import_bundle(destination, home=home)


def test_import_does_not_trust_completed_flags_in_the_received_manifest(campaign, tmp_path):
    home = campaign["home"]
    _, destination, manifest = exported(campaign, tmp_path)
    manifest["files"].append(
        dict(
            path="state/missing.bin", bytes=1, sha256="0" * 64, part=manifest["parts"][0], done=True
        )
    )
    manifest["total_bytes"] += 1
    (destination / data.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no trae 1 archivos"):
        data.import_bundle(destination, home=home)


def test_import_rejects_repeated_archive_members(campaign, tmp_path):
    _, destination, manifest = exported(campaign, tmp_path)
    part = destination / manifest["parts"][0]
    record = manifest["files"][0]
    with tarfile.open(part, "a") as archive:
        archive.add(campaign["home"] / record["path"], arcname=record["path"])
    with pytest.raises(ValueError, match="repetido"):
        data.import_bundle(destination, home=campaign["home"])


@pytest.mark.parametrize("part", ["../outside.tar", "/outside.tar", "nested/part.tar", ""])
def test_import_rejects_parts_outside_the_bundle_before_reading_them(campaign, tmp_path, part):
    _, destination, manifest = exported(campaign, tmp_path)
    manifest["parts"][0] = part
    for record in manifest["files"]:
        record["part"] = part
    (destination / data.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="partes no válidas"):
        data.import_bundle(destination, home=campaign["home"])


def test_import_uses_a_private_temporary_file_without_following_an_existing_link(
    campaign, tmp_path
):
    _, destination, _ = exported(campaign, tmp_path)
    victim = campaign["home"] / "state/edition/samples/US/AA/samples.parquet"
    expected = victim.read_bytes()
    victim.unlink()
    unrelated = write(tmp_path / "unrelated.txt", "contenido ajeno")
    original = unrelated.read_bytes()
    link = victim.with_name(f".{victim.name}.importing")
    link.symlink_to(unrelated)
    assert data.import_bundle(destination, home=campaign["home"])["restored"] == 1
    assert victim.read_bytes() == expected and unrelated.read_bytes() == original
    assert link.is_symlink()


def test_parts_include_tar_headers_padding_and_refuse_an_indivisible_file(tmp_path):
    home, destination = tmp_path / "home", tmp_path / "disk"
    destination.mkdir()
    large = write(home / "data.bin", "x", repeat=1024**2 - 512)
    with pytest.raises(ValueError, match="no cabe en una parte"):
        data.export_bundle({large: digest(large)}, destination, home=home, part_bytes=1024**2)
    assert not list(destination.iterdir())


def test_export_refuses_insufficient_space_before_creating_an_archive(
    campaign, tmp_path, monkeypatch
):
    files = data.campaign_files(campaign["declaration"], campaign["place"])
    destination = tmp_path / "disk"
    destination.mkdir()
    monkeypatch.setattr(shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100, 99, 1))
    with pytest.raises(ValueError, match="espacio libre"):
        data.export_bundle(files, destination, home=campaign["home"])
    assert not list(destination.iterdir())
