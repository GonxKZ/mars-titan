"""Promoción de la criba: regla fijada de antemano y edición del plan del relevo."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


promotion = load("dgx_campaign_promotion", ROOT / "scripts/dgx_campaign_promotion.py")
states = load("screening_states", Path(__file__).with_name("screening_states.py"))
CONTROLS = dict(
    v_buena="base",
    v_regular="base",
    v_mala="base",
    v_escalar="base",
    v_ic="v_escalar",
    v_modalidad="base",
)


def origin(rows):
    """Informe mínimo de un origen: (media de decisión, t) de cada variante frente a su control."""
    models = {name: dict(tier="core" if name == "base" else name) for name in ("base", *rows)}
    contrasts = [
        dict(
            model=name,
            control=CONTROLS[name],
            decision=dict(mean=mean, t=t),
            evaluation=dict(mean=mean / 2, t=t / 2),
        )
        for name, (mean, t) in rows.items()
    ]
    return dict(models=models, contrasts=contrasts)


RESULTS = dict(
    v_buena=(0.010, 3.5),
    v_regular=(0.004, 1.2),
    v_mala=(-0.008, -2.5),
    v_escalar=(-0.001, -0.3),
    v_ic=(0.006, 2.4),
    v_modalidad=(-0.004, -1.0),
)


def test_the_rule_keeps_the_best_variants_the_modality_contrasts_and_their_controls():
    decision = promotion.decide(
        origin(RESULTS), CONTROLS, ["v_modalidad", "no_esta"], keep=2, gain=0.003, t_value=2.0
    )
    assert [row["model"] for row in decision["ranked"]] == [
        "v_buena",
        "v_ic",
        "v_regular",
        "v_escalar",
        "v_modalidad",
        "v_mala",
    ]
    # Siguen las dos mejores, el contraste de modalidad y el control de la que lo necesita.
    assert decision["kept_tiers"] == ["v_buena", "v_escalar", "v_ic", "v_modalidad"]
    # Se promueve lo que supera el umbral con t suficiente, con su control.
    assert decision["promoted_models"] == ["v_buena", "v_ic"]
    assert decision["promoted_tiers"] == ["v_buena", "v_escalar", "v_ic"]
    assert decision["assessed_tiers"] == sorted(RESULTS)
    none = promotion.decide(origin(RESULTS), CONTROLS, [], keep=1, gain=0.02, t_value=2.0)
    assert none["promoted_tiers"] == [] and none["kept_tiers"] == ["v_buena"]


def plan_lines(us, joint):
    tiers = " ".join(f"--tier {name}" for name in ("core", "controls", *sorted(RESULTS)))
    one, other = f"/copia|{us}|/vistas|d.json|", f"/copia|{joint}|/vistas|j.json|"
    return [
        "# plan de prueba",
        f"{one}4 --seed 42 --origin 2013 --no-preflight {tiers}",
        f"orden|/herramientas/promocion|--state {us}",
        f"prepara|/copia|{us}|receta.json|2014",
        f"{one}4 --seed 42 --origin 2014 --no-preflight {tiers}",
        f"limpia|{us}/packed-inputs/2014",
        f"{one}4 --seed 42 --origin 2023 --no-preflight --tier core --tier controls",
        f"{one}3 --seed 42 --origin 2023 --no-preflight --tier networks",
        f"{one}4 --seed 43 --seed 44 --origin 2023 --no-preflight --tier core",
        f"{other}4 --seed 42 --origin 2014 --no-preflight {tiers}",
        f"{other}4 --seed 42 --origin 2023 --no-preflight --tier core",
        "",
    ]


def test_the_edition_only_touches_later_phases_of_the_declared_states_and_origins():
    decision = promotion.decide(
        origin(RESULTS), CONTROLS, ["v_modalidad"], keep=2, gain=0.003, t_value=2.0
    )
    lines = plan_lines("/estado/us", "/estado/conjunta")
    prune = {"/estado/us": {2013, 2014}}
    promote = {"/estado/us": {2023}, "/estado/conjunta": {2023}}
    edited, changes = promotion.edit_plan(lines, 2, decision, prune, promote)
    # La fase ya recorrida (2013) y las que no son de ejecución quedan como estaban.
    assert edited[:4] == lines[:4] and edited[5] == lines[5]
    assert edited[4].endswith(
        "--no-preflight --tier core --tier controls --tier v_buena --tier v_escalar --tier v_ic "
        "--tier v_modalidad"
    )
    # En los orígenes finales entran las promovidas, solo en la fase del núcleo y la semilla 42.
    assert edited[6].endswith(
        "--tier core --tier controls --tier v_buena --tier v_escalar --tier v_ic"
    )
    assert edited[7] == lines[7] and edited[8] == lines[8]
    # La conjunta no se poda (no está declarada en la criba) y sí recibe las promovidas.
    assert edited[9] == lines[9]
    assert edited[10].endswith("--tier core --tier v_buena --tier v_escalar --tier v_ic")
    assert [(c["phase"], c["origin"]) for c in changes] == [(4, 2014), (6, 2023), (10, 2023)]
    assert changes[0]["removed"] == ["v_mala", "v_regular"] and changes[0]["added"] == []
    assert all(promotion.valid(line) for line in edited if promotion.is_phase(line))


@pytest.fixture
def campaign(tmp_path):
    state = tmp_path / "estado"
    states.write_model(state, "base", 2013, 0.15, seed=1)
    states.write_model(state, "v_buena", 2013, 0.45, seed=2, tier="v_buena")
    states.write_model(state, "v_mala", 2013, 0.0, seed=3, tier="v_mala")
    states.write_model(state, "v_modalidad", 2013, 0.15, seed=4, tier="v_modalidad")
    contrasts = tmp_path / "contrastes.json"
    contrasts.write_text(
        json.dumps(
            dict(
                controls=dict(v_buena="base", v_mala="base", v_modalidad="base"),
                always_kept=["v_modalidad"],
            )
        )
    )
    tiers = "--tier core --tier v_buena --tier v_mala --tier v_modalidad"
    plan = tmp_path / "plan.txt"
    run = f"/copia|{state}|/vistas|d.json|4 --seed 42"
    lines = [
        f"{run} --origin 2013 --no-preflight {tiers}",
        f"orden|/herramientas/promocion|--state {state}",
        f"{run} --origin 2014 --no-preflight {tiers}",
        f"{run} --origin 2023 --no-preflight --tier core",
        "",
    ]
    plan.write_text("\n".join(lines))
    return state, contrasts, plan


def arguments(state, contrasts, plan, output, **changes):
    values = dict(keep="1", phase="2")
    values.update(changes)
    paths = dict(state=state, contrasts=contrasts, output=output, plan=plan)
    command = ["--year", "2013", "--phase", values["phase"], "--keep", values["keep"]]
    for name, path in paths.items():
        command += [f"--{name}", str(path)]
    return [*command, "--prune", f"{state}:2014", "--promote", f"{state}:2023"]


def test_the_command_writes_the_records_and_edits_the_plan_once(campaign, tmp_path):
    state, contrasts, plan = campaign
    before = plan.read_text().split("\n")
    output = tmp_path / "registros"
    assert promotion.main(arguments(state, contrasts, plan, output)) == 0
    after = plan.read_text().split("\n")
    assert after[:2] == before[:2]
    assert after[2].endswith("--no-preflight --tier core --tier v_buena --tier v_modalidad")
    assert after[3].endswith("--tier core --tier v_buena")
    document = json.loads((output / "promotion.json").read_text())
    assert document["decision"]["promoted_models"] == ["v_buena"] and "error" not in document
    assert [change["phase"] for change in document["changes"]] == [3, 4]
    assert (output / "plan-anterior.txt").read_text().split("\n") == before
    assert (output / "report.md").is_file() and json.loads((output / "report.json").read_text())
    assert "Promovidas a los orígenes finales: `v_buena`." in (output / "promotion.md").read_text()
    # Repetida (tras una pausa del relevo), no cambia nada más.
    assert promotion.main(arguments(state, contrasts, plan, output)) == 0
    assert plan.read_text().split("\n") == after
    assert json.loads((output / "promotion.json").read_text())["changes"] == []


def test_a_failure_leaves_the_plan_untouched_and_recorded(campaign, tmp_path):
    state, contrasts, plan = campaign
    before = plan.read_text()
    output = tmp_path / "registros"
    # Sin variantes terminadas no hay nada que decidir.
    empty = tmp_path / "vacio"
    (empty / "queue").mkdir(parents=True)
    assert promotion.main(arguments(empty, contrasts, plan, output)) == 1
    document = json.loads((output / "promotion.json").read_text())
    assert "Ninguna variante ha terminado" in document["error"] and plan.read_text() == before
    assert "No se ha tocado el plan" in (output / "promotion.md").read_text()
    # Una fase que no corresponde al plan tampoco lo cambia.
    assert promotion.main(arguments(state, contrasts, plan, output, phase="0")) == 0
    late = promotion.main(arguments(state, contrasts, plan, tmp_path / "otros", phase="9"))
    assert (
        late == 0 and json.loads((tmp_path / "otros/promotion.json").read_text())["changes"] == []
    )


@pytest.mark.parametrize(
    ("line", "accepted"),
    [
        ("limpia|/estado/packed-inputs/2013", True),
        ("limpia|/estado/runs", False),
        ("prepara|/copia|/estado|configs/receta.json|2023", True),
        ("prepara|/copia|/estado|configs/receta.json|23", False),
        ("orden|/herramientas/informe|--state /estado --year 2013", True),
        ("orden|informe|--state /estado", False),
        ("/copia|/estado|/vistas|d.json|4 --seed 42 --origin 2013", True),
        ("/copia|/estado|/vistas|d.json|9 --seed 42", False),
        ("/copia|/estado|/vistas|4 --seed 42", False),
    ],
)
def test_the_phase_forms_are_the_ones_the_relay_accepts(line, accepted):
    assert promotion.valid(line) is accepted


def test_another_group_of_tiers_is_assessed_with_its_own_prefix():
    models = dict(
        base=dict(tier="core"),
        congelada=dict(tier="controls"),
        base_rama=dict(tier="r3_branch"),
        congelada_rama=dict(tier="r3_branch"),
        lector_rama=dict(tier="r3_branch_more"),
        v_buena=dict(tier="v_buena"),
    )
    controls = dict(
        base_rama="base", congelada_rama="congelada", lector_rama="base", v_buena="base"
    )
    rows = dict(
        base_rama=(0.012, 4.0),
        congelada_rama=(0.001, 0.4),
        lector_rama=(0.002, 0.9),
        v_buena=(0.010, 3.5),
    )
    contrasts = [
        dict(
            model=name,
            control=controls[name],
            decision=dict(mean=mean, t=t),
            evaluation=dict(mean=mean, t=t),
        )
        for name, (mean, t) in rows.items()
    ]
    report = dict(models=models, contrasts=contrasts)
    options = dict(keep=12, gain=0.003, t_value=2.0)
    decision = promotion.decide(report, controls, [], prefixes=("r3_",), **options)
    # Solo se valoran los niveles del prefijo; basta un modelo que gane para promover su nivel.
    assert decision["assessed_tiers"] == ["r3_branch", "r3_branch_more"]
    assert decision["promoted_models"] == ["base_rama"] and decision["promoted_tiers"] == [
        "r3_branch"
    ]
    assert promotion.decide(report, controls, [], **options)["promoted_tiers"] == ["v_buena"]
    one = "/copia|/estado/us|/vistas|d.json|"
    lines = [
        f"{one}4 --seed 42 --origin 2013 --no-preflight --tier core --tier r3_branch",
        "orden|/herramientas/promocion|--state /estado/us",
        f"{one}4 --seed 42 --origin 2014 --no-preflight --tier core --tier controls --tier v_buena",
        f"{one}3 --seed 42 --origin 2014 --no-preflight --tier networks",
    ]
    edited, changes = promotion.edit_plan(lines, 2, decision, {}, {"/estado/us": {2014}}, ("r3_",))
    assert edited[2].endswith("--tier core --tier controls --tier v_buena --tier r3_branch")
    assert edited[:2] == lines[:2] and edited[3] == lines[3]
    assert changes == [
        dict(phase=3, state="/estado/us", origin=2014, removed=[], added=["r3_branch"])
    ]
    # La poda con ese prefijo deja el nivel mejor valorado y no toca las variantes de la criba.
    best = promotion.decide(
        report, controls, [], prefixes=("r3_",), keep=1, gain=0.003, t_value=2.0
    )
    assert best["kept_tiers"] == ["r3_branch"]
    pruned, _ = promotion.edit_plan(
        [lines[0], lines[1], lines[2] + " --tier r3_branch_more"],
        2,
        best,
        {"/estado/us": {2014}},
        {},
        ("r3_",),
    )
    assert pruned[2].endswith("--tier core --tier controls --tier v_buena")


def test_promotion_uses_hac_t_and_preserves_plan_cli_flags(tmp_path):
    """Preserva banderas CLI (Fase 11 y 17) y rechaza ruido AR(1) bajo t HAC."""
    from datetime import UTC, datetime, timedelta

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    state = tmp_path / "estado_hac"
    rng = np.random.default_rng(123)
    # Construimos un control y una variante r4_ruido_ar1 cuyas diferencias de IC en calibración
    # son un proceso AR(1) con phi=0.85 que supera t_iid >= 2.5 pero cae por debajo de t_hac < 2.0.
    screening = load("dgx_screening_report", ROOT / "scripts/dgx_screening_report.py")
    n_cal = 120
    n_assets = 40
    # Buscamos una serie AR(1) determinista que cumpla t_iid > 2.5 y t_hac < 2.0.
    ar_diff = None
    for trial in range(500):
        gen = np.random.default_rng(1000 + trial)
        eps = gen.normal(0.0, 0.05, size=n_cal)
        s = np.empty(n_cal, dtype=np.float64)
        s[0] = eps[0]
        for i in range(1, n_cal):
            s[i] = 0.85 * s[i - 1] + eps[i]
        s += 0.025
        m = float(np.mean(s))
        t_iid_val = m / (float(np.std(s, ddof=1)) / (n_cal**0.5))
        t_hac_val = screening.newey_west_t(s)
        if m >= 0.003 and t_iid_val >= 2.5 and t_hac_val is not None and 0.0 < t_hac_val < 2.0:
            ar_diff = s
            break
    assert ar_diff is not None

    # Escribimos predicciones sintéticas donde el IC de r4_ruido_ar1 - control reproduce ar_diff.
    def _write_exact_ic(model_name, tier, ic_by_split):
        run = state / "runs" / model_name / "2013" / "seed-42"
        run.mkdir(parents=True, exist_ok=True)
        for split_idx, (split, ics) in enumerate(ic_by_split.items()):
            rows = dict(sample_id=[], market=[], prediction_at=[], target=[], prediction=[])
            for sess_i, target_ic in enumerate(ics):
                instant = datetime(2012, 1, 1, 20, 5, tzinfo=UTC) + timedelta(
                    days=split_idx * 150 + sess_i
                )
                # Objetivo determinista creciente 0..n_assets-1; mezclamos con ruido ortogonal para
                # lograr correlación de rangos controlada.
                x = np.linspace(-1.0, 1.0, n_assets)
                orth = np.sin(np.arange(n_assets) * 1.7 + sess_i)
                orth = orth - (np.dot(orth, x) / np.dot(x, x)) * x
                orth = orth / np.linalg.norm(orth)
                x_u = x / np.linalg.norm(x)
                r = float(np.clip(target_ic, -0.95, 0.95))
                pred = r * x_u + ((1.0 - r * r) ** 0.5) * orth
                for a in range(n_assets):
                    rows["sample_id"].append(f"US/A{a:03d}/{instant.timestamp():.0f}")
                    rows["market"].append("US")
                    rows["prediction_at"].append(instant)
                    rows["target"].append(float(x[a]))
                    rows["prediction"].append(float(pred[a]))
            table = pa.table(rows).cast(
                pa.schema(
                    [
                        ("sample_id", pa.string()),
                        ("market", pa.string()),
                        ("prediction_at", pa.timestamp("us", tz="UTC")),
                        ("target", pa.float64()),
                        ("prediction", pa.float64()),
                    ]
                )
            )
            pq.write_table(table, run / f"{split}-predictions.parquet")
        queue = state / "queue"
        queue.mkdir(exist_ok=True)
        (queue / f"{model_name}--2013--seed-42.json").write_text(
            json.dumps(
                dict(
                    status="completed",
                    model=model_name,
                    tier=tier,
                    year=2013,
                    seed=42,
                    attempts=[dict(seconds=60.0)],
                    result=dict(best_epoch=1, best_score=0.49),
                )
            )
        )

    base_cal = np.full(n_cal, 0.10)
    base_val = np.full(20, 0.10)
    base_eval = np.full(20, 0.10)
    _write_exact_ic(
        "control", "core", dict(validation=base_val, calibration=base_cal, evaluation=base_eval)
    )
    _write_exact_ic(
        "r4_ruido_ar1",
        "r4_ruido",
        dict(validation=base_val, calibration=base_cal + ar_diff, evaluation=base_eval),
    )

    contrasts = tmp_path / "r4_contrasts.json"
    contrasts.write_text(json.dumps(dict(controls=dict(r4_ruido_ar1="control"), always_kept=[])))
    plan = tmp_path / "plan_r4.txt"
    run_line = f"/copia|{state}|/vistas|d.json|4 --seed 42"
    plan.write_text(
        "\n".join(
            [
                f"{run_line} --origin 2013 --no-preflight --tier core --tier r4_ruido",
                f"orden|/herramientas/promocion|--state {state}",
                f"{run_line} --origin 2014 --no-preflight --tier core",
                "",
            ]
        )
    )
    out_r4 = tmp_path / "out_r4"
    # Invocación exacta de Fase 11 con --candidates r4_ y --promote.
    rc = promotion.main(
        [
            "--state",
            str(state),
            "--year",
            "2013",
            "--contrasts",
            str(contrasts),
            "--output",
            str(out_r4),
            "--plan",
            str(plan),
            "--phase",
            "2",
            "--promote",
            f"{state}:2014",
            "--candidates",
            "r4_",
            "--decision-split",
            "calibration",
        ]
    )
    assert rc == 0
    doc = json.loads((out_r4 / "promotion.json").read_text())
    row = doc["decision"]["ranked"][0]
    assert row["model"] == "r4_ruido_ar1"
    assert row["decision_t_iid"] >= 2.5
    assert row["decision_t"] < 2.0
    assert row["p_bh"] is not None
    # Rechazada por HAC t < 2.0 aunque habría pasado con t_iid >= 2.5.
    assert doc["decision"]["promoted_models"] == []
    assert doc["rule"]["decision_splits"] == ["calibration"]
    assert doc["rule"]["candidates"] == ["r4_"]
    _ = rng
