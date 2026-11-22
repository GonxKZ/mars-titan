"""Fallos de las herramientas que señaló la auditoría del tercer enjambre (22 de noviembre de 2026).

Cada prueba falla con el código anterior a los arreglos: la equivalencia de Colab declarada con un
error no finito o con tramos que se cancelan, predicciones NaN puntuadas como aciertos, el espejo
de Drive destruido al reescribir, un origen de 2024 admitido por el guion de Colab, rutas relativas
que se rompen al cambiar de carpeta y una búsqueda que no se recupera de una interrupción.
"""

import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
START = 1_357_160_700_000_000
DAY = 86_400_000_000


def resolve_path(relative):
    """La ruta de un fichero del repositorio; las pruebas lo nombran desde la raíz `repo-v11/`."""
    return ROOT / relative.removeprefix("repo-v11/")


def load_script(relative, module_name):
    """Cargar un guion independiente de `scripts/` como módulo."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, resolve_path(relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("market", pa.string()),
            ("prediction_at", pa.timestamp("us", tz="UTC")),
            ("target", pa.float64()),
            ("prediction", pa.float64()),
        ]
    )
    table = (
        pa.Table.from_pylist(rows, schema=schema)
        if rows
        else pa.Table.from_pydict(
            {field.name: pa.array([], type=field.type) for field in schema}, schema=schema
        )
    )
    pq.write_table(table, path)


def _make_executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# H05 y H06: material/repo-v11/scripts/colab_parity.py
# ---------------------------------------------------------------------------


def test_h05_colab_parity_nan_or_inf_se_never_equivalent():
    parity = load_script("repo-v11/scripts/colab_parity.py", "audit_colab_parity_h05a")
    # Una única sesión produce standard_error = NaN e intervalo [-inf, +inf].
    se_one = parity.hac_standard_error(np.array([0.0001]), ["US"])
    assert math.isnan(se_one)
    assert parity.tost(0.0001, se_one, 0.002) == 1.0
    label_one, _ = parity.verdict(0.0001, se_one, 0.002)
    assert label_one == "no concluyente"
    # Error estándar infinito tampoco puede declararse equivalente.
    assert parity.tost(0.0, float("inf"), 0.002) == 1.0
    label_inf, _ = parity.verdict(0.0, float("inf"), 0.002)
    assert label_inf == "no concluyente"
    # Media NaN o margen no positivo tampoco.
    assert parity.tost(float("nan"), 0.0001, 0.002) == 1.0
    label_nan, _ = parity.verdict(float("nan"), 0.0001, 0.002)
    assert label_nan == "no concluyente"
    assert parity.tost(0.0, 0.0001, 0.0) == 1.0
    assert parity.verdict(0.0, 0.0001, 0.0)[0] == "no concluyente"


def test_h05_colab_parity_split_cancellation_cannot_report_equivalent():
    parity = load_script("repo-v11/scripts/colab_parity.py", "audit_colab_parity_h05b")
    # Validación tiene sesgo +0.005 (distinto) y evaluación -0.005 (distinto), > 2.5x margen.
    # En conjunto sus medias se cancelan a ~0 y el centrado por bloque borra el salto del SE.
    v_arr = np.full(30, 0.005) + np.linspace(-1e-4, 1e-4, 30)
    c_arr = np.full(20, 0.0) + np.linspace(-1e-4, 1e-4, 20)
    e_arr = np.full(30, -0.005) + np.linspace(-1e-4, 1e-4, 30)
    rep = {
        "splits": {
            "validation": {
                "same_rows": True,
                "same_targets": True,
                "keys": [f"US|{i}" for i in range(30)],
                "difference": v_arr.tolist(),
            },
            "calibration": {
                "same_rows": True,
                "same_targets": True,
                "keys": [f"US|{i}" for i in range(20)],
                "difference": c_arr.tolist(),
            },
            "evaluation": {
                "same_rows": True,
                "same_targets": True,
                "keys": [f"US|{i}" for i in range(30)],
                "difference": e_arr.tolist(),
            },
        }
    }
    summary = parity.summarize([rep], margin=0.002)
    assert summary["splits"]["validation"]["verdict"] == "distinto"
    assert summary["splits"]["evaluation"]["verdict"] == "distinto"
    assert summary["verdict"] == "distinto"


def test_h06_colab_parity_rejects_nan_predictions_and_checks_alignment(tmp_path):
    parity = load_script("repo-v11/scripts/colab_parity.py", "audit_colab_parity_h06")
    # Un modelo que emite todo NaN no puede obtener IC = +1.0.
    nan_corr = parity.rank_correlation(np.full(40, np.nan), np.arange(40, dtype=float))
    assert math.isnan(nan_corr)
    partial_nan = np.arange(40, dtype=float)
    partial_nan[0] = np.nan
    assert math.isnan(parity.rank_correlation(partial_nan, np.arange(40, dtype=float)))
    assert math.isnan(parity.rank_correlation(np.arange(40, dtype=float), partial_nan))

    # compare_runs con el mismo sample_id pero distinto market o prediction_at debe marcar
    # same_rows=False (en lugar de lanzar KeyError al indexar ic_b[key]).
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    for split in parity.SPLITS:
        rows_a, rows_b = [], []
        for i in range(35):
            rows_a.append(
                dict(
                    sample_id=f"S{i:03d}",
                    market="US",
                    prediction_at=START,
                    target=float(i),
                    prediction=float(i),
                )
            )
            rows_b.append(
                dict(
                    sample_id=f"S{i:03d}",
                    market="CN" if split == "evaluation" else "US",
                    prediction_at=START,
                    target=float(i),
                    prediction=float(i),
                )
            )
        _write_predictions(run_a / f"{split}-predictions.parquet", rows_a)
        _write_predictions(run_b / f"{split}-predictions.parquet", rows_b)
    compared = parity.compare_runs(run_a, run_b)
    assert compared["splits"]["evaluation"]["same_rows"] is False


# ---------------------------------------------------------------------------
# H09: material/repo-v11/scripts/colab_sync.py
# ---------------------------------------------------------------------------


def test_h09_colab_sync_preserves_existing_mirror_on_rewrite_and_holds_dependent_run_json(
    tmp_path, monkeypatch
):
    sync = load_script("repo-v11/scripts/colab_sync.py", "audit_colab_sync_h09")
    state = tmp_path / "state"
    mirror = tmp_path / "mirror"
    ckpts = Path("runs/m/2013/seed-42/fit/checkpoints")
    (state / "queue").mkdir(parents=True)
    (state / "queue/m--2013--seed-42.json").write_text('{"status": "running"}', encoding="utf-8")
    (state / ckpts).mkdir(parents=True)
    (state / ckpts / "state-aaa.pt").write_bytes(b"A" * 10)
    (state / ckpts / "latest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                latest=[dict(name="state-aaa.pt", size=10, global_step=1)],
                best=None,
                pinned=[],
            )
        ),
        encoding="utf-8",
    )
    (state / "runs/m/2013/seed-42/fit/run.json").write_text(
        json.dumps(dict(status="running", step=1)), encoding="utf-8"
    )
    (state / "authorization.json").write_text('{"version": 1}', encoding="utf-8")

    # Primera sincronización limpia.
    first = sync.push(state, mirror)
    assert first["held"] == [] and (mirror / "authorization.json").read_text() == '{"version": 1}'

    # 1) Si authorization.json cambia en origen MIENTRAS se copia en el segundo push, la copia
    # válida anterior del espejo NO puede ser borrada por unlink().
    target = state / "authorization.json"
    target.write_text('{"version": 2}', encoding="utf-8")
    os.utime(target, ns=(2_000_000_000, 2_000_000_000))
    orig_write = sync.write_atomic

    def rewriting(destination, data, mtime_ns=None):
        orig_write(destination, data, mtime_ns)
        if destination.name == "authorization.json":
            target.write_text('{"version": 3_concurrent}', encoding="utf-8")
            os.utime(target, ns=(3_000_000_000, 3_000_000_000))

    monkeypatch.setattr(sync, "write_atomic", rewriting)
    second = sync.push(state, mirror)
    monkeypatch.setattr(sync, "write_atomic", orig_write)
    assert "authorization.json" in second["retried_later"]
    assert (mirror / "authorization.json").is_file(), (
        "Se destruyó la copia válida previa del espejo"
    )
    assert (mirror / "authorization.json").read_text(encoding="utf-8") == '{"version": 1}'

    # 2) Si latest.json apunta a un estado inexistente (retenido), fit/run.json actualizado para ese
    # paso NO debe publicarse por delante de latest.json.
    (state / ckpts / "latest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                latest=[dict(name="state-missing.pt", size=12, global_step=2)],
                best=None,
                pinned=[],
            )
        ),
        encoding="utf-8",
    )
    (state / "runs/m/2013/seed-42/fit/run.json").write_text(
        json.dumps(dict(status="running", step=2)), encoding="utf-8"
    )
    third = sync.push(state, mirror)
    assert (ckpts / "latest.json").as_posix() in third["held"]
    assert json.loads((mirror / "runs/m/2013/seed-42/fit/run.json").read_text())["step"] == 1


# ---------------------------------------------------------------------------
# H10: material/repo-v11/scripts/colab_origin.sh y colab_bootstrap.sh
# ---------------------------------------------------------------------------


def test_h10_colab_origin_blocks_2024_resolves_relative_paths_and_rejects_globs(tmp_path):
    script = resolve_path("repo-v11/scripts/colab_origin.sh")
    repo = tmp_path / "repo"
    (repo / "src/mars_titan").mkdir(parents=True)
    (repo / "decl.json").write_text("{}", encoding="utf-8")
    (repo / "scripts").mkdir(parents=True)
    log_file = tmp_path / "launcher.log"
    _make_executable(
        repo / "scripts/colab_campaign.sh",
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$PWD" "$MARS_TITAN_COLAB_PYTHON" "$@" > "{log_file}"\n',
    )
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "estado").mkdir()
    (caller / "vistas").mkdir()
    (caller / "archivo_glob.txt").write_text("x", encoding="utf-8")
    fake_py = _make_executable(caller / "mi_python", "#!/usr/bin/env bash\nexit 0\n")

    env = dict(
        os.environ,
        MARS_TITAN_SITE="test-colab",
        MARS_TITAN_COLAB_PYTHON="./mi_python",
    )
    # 1) --origen 2024 DEBE ser rechazado (2024 está sellado).
    res_2024 = subprocess.run(
        [
            "bash",
            str(script),
            "estado",
            "--repo",
            str(repo),
            "--estado",
            str(caller / "estado"),
            "--vistas",
            str(caller / "vistas"),
            "--declaracion",
            "decl.json",
            "--origen",
            "2024",
            "--ensayo",
        ],
        cwd=caller,
        env=env,
        capture_output=True,
        text=True,
    )
    assert res_2024.returncode != 0, f"colab_origin.sh aceptó --origen 2024: {res_2024.stdout}"

    # 2) Rutas relativas en --estado, --vistas y MARS_TITAN_COLAB_PYTHON deben resolverse al
    # directorio del llamante antes de `cd "$repo"`.
    res_rel = subprocess.run(
        [
            "bash",
            str(script),
            "estado",
            "--repo",
            str(repo),
            "--estado",
            "./estado",
            "--vistas",
            "./vistas",
            "--declaracion",
            "decl.json",
            "--origen",
            "2023",
        ],
        cwd=caller,
        env=env,
        capture_output=True,
        text=True,
    )
    assert res_rel.returncode == 0, res_rel.stderr
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert Path(lines[1]) == fake_py.resolve()
    state_arg = lines[lines.index("--state") + 1]
    views_arg = lines[lines.index("--views") + 1]
    assert Path(state_arg) == (caller / "estado").resolve()
    assert Path(views_arg) == (caller / "vistas").resolve()

    # 3) --semillas "*" no puede expandir globs del directorio de trabajo ni aceptar no enteros.
    res_glob = subprocess.run(
        [
            "bash",
            str(script),
            "lanza",
            "--repo",
            str(repo),
            "--estado",
            str(caller / "estado"),
            "--vistas",
            str(caller / "vistas"),
            "--declaracion",
            "decl.json",
            "--origen",
            "2023",
            "--semillas",
            "*",
            "--ensayo",
        ],
        cwd=caller,
        env=env,
        capture_output=True,
        text=True,
    )
    assert res_glob.returncode != 0, f"Aceptó glob en --semillas: {res_glob.stdout}"


def test_h10_colab_bootstrap_relative_bundle_missing_nativo_and_relative_trabajo(tmp_path):
    script = resolve_path("repo-v11/scripts/colab_bootstrap.sh")
    # Crear un repositorio Git real y un bundle para probar paso_codigo con ruta relativa.
    origin_repo = tmp_path / "origin_repo"
    origin_repo.mkdir()
    git_env = dict(
        os.environ,
        GIT_AUTHOR_NAME="Audit",
        GIT_AUTHOR_EMAIL="audit@example.invalid",
        GIT_COMMITTER_NAME="Audit",
        GIT_COMMITTER_EMAIL="audit@example.invalid",
    )
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=origin_repo, env=git_env, check=True)
    (origin_repo / "README.md").write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=origin_repo, env=git_env, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "v1"], cwd=origin_repo, env=git_env, check=True)
    commit1 = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=origin_repo, text=True
    ).strip()
    caller = tmp_path / "caller"
    caller.mkdir()
    bundle = caller / "repo.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), "HEAD"], cwd=origin_repo, env=git_env, check=True
    )

    # Primer codigo (clone) y segundo codigo (fetch con --bundle relativo y --trabajo relativo).
    cmd_code = [
        "bash",
        str(script),
        "codigo",
        "--trabajo",
        "./trabajo",
        "--bundle",
        "./repo.bundle",
        "--commit",
        commit1,
    ]
    r1 = subprocess.run(cmd_code, cwd=caller, env=git_env, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(cmd_code, cwd=caller, env=git_env, capture_output=True, text=True)
    assert r2.returncode == 0, f"Re-ejecución con --bundle relativo falló: {r2.stderr}"

    # paso_comprobar sin $trabajo/nativo y con --trabajo relativo: usando un stub de `uv`.
    repo = caller / "trabajo/repo"
    (repo / ".venv").mkdir(parents=True)
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    uv_log = tmp_path / "uv.log"
    _make_executable(
        stub_bin / "uv",
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "{uv_log}"\nexit 0\n',
    )
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    env_stub = dict(os.environ, HOME=str(fake_home), PATH=f"{stub_bin}:{os.environ['PATH']}")
    r3 = subprocess.run(
        ["bash", str(script), "comprobar", "--trabajo", "./trabajo"],
        cwd=caller,
        env=env_stub,
        capture_output=True,
        text=True,
    )
    assert r3.returncode == 0, f"paso_comprobar falló sin nativo/: {r3.stderr}"
    logged = uv_log.read_text(encoding="utf-8").splitlines()
    out_arg = logged[logged.index("--output") + 1]
    assert Path(out_arg) == (caller / "trabajo/equipo.json").resolve()


# ---------------------------------------------------------------------------
# H12: material/repo-v11/scripts/campaign_search.py
# ---------------------------------------------------------------------------


def test_h12_campaign_search_markdown_ordinal_none_text_bool_and_schema_version():
    search = load_script("repo-v11/scripts/campaign_search.py", "audit_campaign_search_h12a")
    # 1) schema_version=True debe ser rechazado (el resto de la declaración es 100% válida).
    manifest = dict(
        schema_version=True,
        kind=search.KIND,
        final_test_access=False,
        study="estudio-a",
        tag="a",
        campaign="camp.json",
        base_model="base",
        year=2013,
        run_seed=42,
        seed=7,
        startup=2,
        confirm=1,
        space=[
            dict(name="recipe.recency_half_life_years", type="ordinal", values=[None, 2.0, 4.0])
        ],
    )
    with pytest.raises(ValueError):
        search.validate(manifest)

    # 2) markdown() con ordinales None, texto y booleanos no debe lanzar TypeError/ValueError.
    doc = dict(
        schema_version=1,
        kind="campaign_search_report_v1",
        final_test_opened=False,
        study="s",
        base_model="base",
        year=2013,
        run_seed=42,
        space=[
            dict(name="recipe.recency_half_life_years", kind="ordinal", values=[None, 2.0, 4.0]),
            dict(
                name="predictor.position", kind="ordinal", values=["after_mac", "after_technical"]
            ),
            dict(name="recipe.grad_clip_by_role", kind="ordinal", values=[False, True]),
        ],
        told=1,
        pending=0,
        failed=0,
        known={
            "base": dict(
                values={
                    "recipe.recency_half_life_years": None,
                    "predictor.position": "after_mac",
                    "recipe.grad_clip_by_role": False,
                },
                validation=dict(mean=0.01),
                calibration=dict(mean=0.01),
            )
        },
        trials=[
            dict(
                number=0,
                generation=0,
                model="s__t000",
                status="told",
                values={
                    "recipe.recency_half_life_years": None,
                    "predictor.position": "after_technical",
                    "recipe.grad_clip_by_role": True,
                },
                validation=dict(mean=0.02),
                calibration=dict(mean=0.02),
                against_base=dict(
                    validation=dict(mean=0.01, t=1.5), calibration=dict(mean=0.01, t=1.2)
                ),
            )
        ],
        candidates=[],
        concordance=None,
        chance_maximum=None,
        recommended=None,
    )
    md = search.markdown(doc)
    assert "after_technical" in md and "none" in md.lower() or "—" in md
    assert "| true |" in md.lower() and "| false |" in md.lower()


def test_h12_campaign_search_tell_recovery_after_crash_and_report_no_nan_json(
    tmp_path, monkeypatch
):
    search = load_script("repo-v11/scripts/campaign_search.py", "audit_campaign_search_h12b")
    declared = dict(
        schema_version=1,
        kind=search.KIND,
        final_test_access=False,
        study="estudio-rec",
        tag="rec",
        campaign="camp.json",
        base_model="base",
        year=2013,
        run_seed=42,
        seed=7,
        startup=2,
        confirm=1,
        space=[dict(name="recipe.learning_rate", type="float", low=1e-4, high=1e-2, log=True)],
    )
    folder = tmp_path / "search"
    state = tmp_path / "state"
    state.mkdir()

    # Pedir 3 ensayos directamente vía Optuna + trials.json (sin depender de un repo completo).
    study = search._study(folder, declared)
    distributions = search._distributions(declared["space"])
    asked = []
    for _ in range(3):
        trial = study.ask(distributions)
        asked.append(
            dict(
                number=trial.number,
                generation=1,
                model=f"base_t{trial.number:03d}",
                recipe=f"t{trial.number:03d}.json",
                values=search._values(declared["space"], trial.params),
                status="asked",
            )
        )
    search._write(folder / "trials.json", dict(schema_version=1, trials=asked, known={}))

    # Simular que todos terminaron con la misma media (empate total -> concordancia indefinida).
    class StubReport:
        @staticmethod
        def completed(st, yr, sd):
            return {e["model"]: {} for e in asked}

        @staticmethod
        def summary(values):
            arr = list(values.values()) if isinstance(values, dict) else list(values)
            return dict(mean=float(np.mean(arr)), se=0.01, t=1.0, sessions=len(arr))

        @staticmethod
        def paired(a, b):
            return dict(mean=0.0, se=0.01, t=0.0, sessions=len(a))

        @staticmethod
        def average_ranks(v):
            return np.full(len(v), (len(v) + 1) / 2.0)

        @staticmethod
        def correlation(a, b):
            return float("nan")

    monkeypatch.setattr(search, "_report_module", lambda: StubReport)
    monkeypatch.setattr(
        search,
        "_measure",
        lambda st, dec, model: {
            "validation": {("US", 1): 0.05},
            "calibration": {("US", 1): 0.05},
        },
    )

    # 1) Interrupción justo después de study.tell() del ensayo 0 y antes de guardar trials.json.
    study = search._study(folder, declared)
    study.tell(asked[0]["number"], 0.05)
    # Ahora trials.json aún tiene el ensayo 0 como "asked". Re-ejecutar tell() debe recuperarse.
    told = search.tell(folder, declared, state)
    assert len(told) == 3

    # 2) report() con medias empatadas NO debe producir NaN (ni JSON no estándar).
    doc = search.report(folder, declared)
    assert doc["concordance"] is None
    serialized = json.dumps(doc, allow_nan=False)
    assert "NaN" not in serialized
