"""La búsqueda sobre la campaña pide ensayos, los vuelve recetas y solo mira validación."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("optuna")

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "campaign_search", ROOT / "scripts" / "campaign_search.py"
)
search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search)

START = 1_357_160_700_000_000
DAY = 86_400_000_000
SESSIONS = {"validation": 30, "calibration": 20, "evaluation": 25}
ASSETS = 40
RATES = [0.0003, 0.001, 0.003, 0.01]


def repository(tmp_path):
    """Un repositorio mínimo: la campaña de partida y la receta del modelo base."""
    root = tmp_path / "repo"
    (root / "configs/titans").mkdir(parents=True)
    (root / "configs/experiments").mkdir(parents=True)
    recipe = dict(
        schema_version=1,
        predictor=dict(hidden_size=64, technical_branch=dict(hidden=128, depth=2)),
        recipe=dict(learning_rate=0.001, weight_decay=0, session_ic_term=dict(weight=3.0)),
        pending=["Sin resultados publicados."],
    )
    (root / "configs/titans/base.json").write_text(json.dumps(recipe))
    campaign = dict(
        schema_version=1,
        kind="comparison_campaign_v1",
        market="US",
        final_test_access=False,
        description="Campaña de partida.",
        seeds=[42, 43, 44],
        tiers=["core", "controls"],
        recipes=dict(core="configs/titans/base.json", other="configs/titans/other.json"),
        models=dict(
            titans_mac=dict(
                machine="mars_titan", recipe="core", runner="fit", tier="core", variant="mac_online"
            ),
            titans_mac_frozen=dict(
                machine="mars_titan",
                recipe="other",
                runner="fit",
                tier="controls",
                variant="mac_frozen",
            ),
        ),
    )
    (root / "configs/experiments/base-campaign.json").write_text(json.dumps(campaign))
    return root


def declaration(**changes):
    document = dict(
        schema_version=1,
        kind="campaign_search_v1",
        study="prueba-1",
        tag="p1",
        final_test_access=False,
        seed=7,
        campaign="configs/experiments/base-campaign.json",
        base_model="titans_mac",
        year=2013,
        run_seed=42,
        startup=8,
        confirm=3,
        space=[
            dict(name="recipe.learning_rate", type="float", low=0.0003, high=0.003, log=True),
            dict(name="recipe.weight_decay", type="ordinal", values=[0, 1e-5, 1e-4, 1e-3]),
            dict(name="predictor.technical_branch.hidden", type="ordinal", values=[64, 128]),
        ],
        known=[
            dict(
                model="titans_mac",
                values={
                    "recipe.learning_rate": 0.001,
                    "recipe.weight_decay": 0,
                    "predictor.technical_branch.hidden": 128,
                },
            )
        ],
    )
    document.update(changes)
    return document


def write_run(state, model, *, signal, noise_seed, status="completed", splits=SESSIONS):
    """Emisiones sintéticas de un trabajo y su registro en la cola."""
    folder = state / "runs" / model / "2013" / "seed-42"
    folder.mkdir(parents=True)
    (state / "queue").mkdir(exist_ok=True)
    record = dict(model=model, status=status)
    (state / "queue" / f"{model}--2013--seed-42.json").write_text(json.dumps(record))
    offset = 0
    for split, count in splits.items():
        rows = []
        for session in range(count):
            labels = np.random.default_rng(10_000 + offset + session).standard_normal(ASSETS)
            noise = np.random.default_rng(noise_seed * 1000 + offset + session).standard_normal(
                ASSETS
            )
            prediction = signal * labels + noise
            instant = START + (offset + session) * DAY
            for asset in range(ASSETS):
                rows.append(
                    dict(
                        sample_id=f"US/A{asset:03d}/{instant}",
                        market="US",
                        prediction_at=instant,
                        target=float(labels[asset]),
                        prediction=float(prediction[asset]),
                    )
                )
        offset += count
        table = pa.Table.from_pylist(rows).cast(
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
        pq.write_table(table, folder / f"{split}-predictions.parquet")


def signal_of(values):
    """Respuesta sintética: mejor cuanto más cerca de 0,001 la tasa y con la rama ancha."""
    distance = abs(np.log10(values["recipe.learning_rate"]) + 3.0)
    return 0.6 - 0.4 * distance + (0.1 if values["predictor.technical_branch.hidden"] == 128 else 0)


def run_batch(state, batch, *, break_evaluation=False):
    for entry in batch:
        write_run(
            state, entry["model"], signal=signal_of(entry["values"]), noise_seed=entry["number"]
        )
        if break_evaluation:
            path = state / "runs" / entry["model"] / "2013" / "seed-42"
            (path / "evaluation-predictions.parquet").write_bytes(b"no es un parquet")


def test_ask_writes_one_recipe_per_trial_and_the_campaign_of_the_batch(tmp_path):
    root = repository(tmp_path)
    study = tmp_path / "study"
    batch = search.ask(study, declaration(), root, 4)
    assert [entry["number"] for entry in batch] == [0, 1, 2, 3]
    base = json.loads((root / "configs/titans/base.json").read_text())
    for entry in batch:
        assert entry["model"] == f"titans_mac_p1t{entry['number']:03d}" and entry["generation"] == 1
        recipe = json.loads((root / entry["recipe"]).read_text())
        values = entry["values"]
        assert 0.0003 <= values["recipe.learning_rate"] <= 0.003
        assert values["recipe.weight_decay"] in (0, 1e-5, 1e-4, 1e-3)
        assert values["predictor.technical_branch.hidden"] in (64, 128)
        assert recipe["recipe"]["learning_rate"] == values["recipe.learning_rate"]
        assert recipe["recipe"]["weight_decay"] == values["recipe.weight_decay"]
        assert (
            recipe["predictor"]["technical_branch"]["hidden"]
            == values["predictor.technical_branch.hidden"]
        )
        # Fuera de los campos declarados, la receta del ensayo es la del modelo base.
        recipe["recipe"].update(learning_rate=0.001, weight_decay=0)
        recipe["predictor"]["technical_branch"]["hidden"] = 128
        note = recipe.pop("pending")
        assert "prueba-1" in note[0] and "2024" in note[1]
        assert recipe == {key: value for key, value in base.items() if key != "pending"}
    campaign = json.loads((root / "configs/experiments/search/prueba-1-g1.json").read_text())
    assert campaign["final_test_access"] is False and campaign["seeds"] == [42, 43, 44]
    assert campaign["tiers"] == ["core", "search_g1"]
    assert set(campaign["models"]) == {"titans_mac", *(entry["model"] for entry in batch)}
    first = campaign["models"][batch[0]["model"]]
    assert first == dict(
        machine="mars_titan",
        recipe="search_p1t000",
        runner="fit",
        tier="search_g1",
        variant="mac_online",
    )
    assert campaign["recipes"]["search_p1t000"] == batch[0]["recipe"]
    assert campaign["recipes"]["core"] == "configs/titans/base.json"
    assert "other" not in campaign["recipes"]
    controls = json.loads(
        (root / "configs/experiments/search/prueba-1-g1-contrasts.json").read_text()
    )["controls"]
    assert controls == {entry["model"]: "titans_mac" for entry in batch}


def test_the_startup_trials_fill_the_space_and_do_not_repeat_between_calls(tmp_path):
    root = repository(tmp_path)
    study = tmp_path / "study"
    batch = search.ask(study, declaration(), root, 3) + search.ask(study, declaration(), root, 5)
    assert [entry["number"] for entry in batch] == list(range(8))
    assert [entry["generation"] for entry in batch] == [1, 1, 1, 2, 2, 2, 2, 2]
    rates = sorted(entry["values"]["recipe.learning_rate"] for entry in batch)
    # Hipercubo latino: un ensayo en cada octavo del intervalo logarítmico de la tasa.
    edges = np.logspace(np.log10(0.0003), np.log10(0.003), 9)
    assert all(
        low <= rate <= high for rate, low, high in zip(rates, edges[:-1], edges[1:], strict=True)
    )
    decays = [entry["values"]["recipe.weight_decay"] for entry in batch]
    assert sorted(decays) == [0, 0, 1e-5, 1e-5, 1e-4, 1e-4, 1e-3, 1e-3]
    widths = [entry["values"]["predictor.technical_branch.hidden"] for entry in batch]
    assert sorted(widths) == [64] * 4 + [128] * 4
    # La segunda tanda acumula a la primera en su campaña.
    campaign = json.loads((root / "configs/experiments/search/prueba-1-g2.json").read_text())
    assert campaign["tiers"] == ["core", "search_g1", "search_g2"]
    assert len(campaign["models"]) == 9


def test_the_same_history_gives_the_same_trials(tmp_path):
    asked = []
    for name in ("a", "b"):
        root = repository(tmp_path / name)
        study, state = tmp_path / name / "study", tmp_path / name / "state"
        first = search.ask(study, declaration(), root, 8)
        run_batch(state, first)
        search.tell(study, declaration(), state)
        second = search.ask(study, declaration(), root, 4)
        asked.append([entry["values"] for entry in first + second])
    assert asked[0] == asked[1]
    assert len({json.dumps(values, sort_keys=True) for values in asked[0]}) == 12


def test_tell_reads_validation_and_calibration_and_never_the_evaluation(tmp_path):
    root = repository(tmp_path)
    study, state = tmp_path / "study", tmp_path / "state"
    batch = search.ask(study, declaration(), root, 4)
    run_batch(state, batch[:2], break_evaluation=True)
    write_run(state, batch[2]["model"], signal=0.5, noise_seed=99, status="running")
    told = search.tell(study, declaration(), state)
    assert [entry["number"] for entry in told] == [0, 1]
    ledger = json.loads((study / "trials.json").read_text())
    by_number = {entry["number"]: entry for entry in ledger["trials"]}
    assert [by_number[number]["status"] for number in range(4)] == ["told"] * 2 + ["asked"] * 2
    table = pq.read_table(
        state / "runs" / batch[0]["model"] / "2013" / "seed-42" / "validation-predictions.parquet"
    ).to_pandas()
    values = [
        rows["prediction"].rank().corr(rows["target"].rank())
        for _, rows in table.groupby("prediction_at")
    ]
    assert by_number[0]["validation"]["mean"] == pytest.approx(np.mean(values), rel=1e-9)
    assert by_number[0]["validation"]["sessions"] == 30
    assert by_number[0]["calibration"]["sessions"] == 20
    assert "evaluation" not in json.dumps(ledger)
    # Lo que ve Optuna es la validación, no la calibración.
    import optuna

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(study / "optuna.journal"))
    )
    trials = optuna.load_study(study_name="prueba-1", storage=storage).trials
    for number in (0, 1):
        assert trials[number].value == by_number[number]["validation"]["mean"]
        assert trials[number].value != by_number[number]["calibration"]["mean"]
    # Volver a leer el mismo estado no cuenta dos veces un ensayo.
    assert search.tell(study, declaration(), state) == []


def test_the_known_models_enter_the_study_once_with_their_measure(tmp_path):
    root = repository(tmp_path)
    study, state = tmp_path / "study", tmp_path / "state"
    batch = search.ask(study, declaration(), root, 2)
    run_batch(state, batch)
    write_run(state, "titans_mac", signal=0.7, noise_seed=500)
    search.tell(study, declaration(), state)
    search.tell(study, declaration(), state)
    ledger = json.loads((study / "trials.json").read_text())
    assert list(ledger["known"]) == ["titans_mac"]
    assert ledger["known"]["titans_mac"]["validation"]["sessions"] == 30
    # Cada ensayo queda contrastado con el modelo base, sesión a sesión.
    paired = ledger["trials"][0]["against_base"]
    assert set(paired) == {"validation", "calibration"}
    assert paired["validation"]["sessions"] == 30 and paired["validation"]["mean"] < 0
    import optuna

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(study / "optuna.journal"))
    )
    trials = optuna.load_study(study_name="prueba-1", storage=storage).trials
    complete = [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE]
    assert len(complete) == 3
    known = [trial for trial in complete if trial.user_attrs.get("known") == "titans_mac"]
    assert len(known) == 1 and known[0].params["recipe.weight_decay"] == 0
    assert known[0].params["recipe.learning_rate"] == 0.001
    assert known[0].value == ledger["known"]["titans_mac"]["validation"]["mean"]
    assert known[0].value != ledger["known"]["titans_mac"]["calibration"]["mean"]


def test_a_failed_trial_stops_counting_and_leaves_the_next_campaign(tmp_path):
    import optuna

    root = repository(tmp_path)
    study, state = tmp_path / "study", tmp_path / "state"
    batch = search.ask(study, declaration(), root, 3)
    run_batch(state, batch[:1])
    search.tell(study, declaration(), state)
    failed = search.fail(study, declaration(), 1, "pérdida no finita")
    assert failed["status"] == "failed" and failed["reason"] == "pérdida no finita"
    # Ni un ensayo ya medido ni uno ya fallido se pueden dar por fallidos.
    for number in (0, 1, 7):
        with pytest.raises(ValueError, match="no está pedido"):
            search.fail(study, declaration(), number, "otra vez")
    later = search.ask(study, declaration(), root, 1)
    campaign = json.loads((root / "configs/experiments/search/prueba-1-g2.json").read_text())
    assert batch[1]["model"] not in campaign["models"]
    assert {batch[0]["model"], batch[2]["model"], later[0]["model"]} <= set(campaign["models"])
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(study / "optuna.journal"))
    )
    trials = optuna.load_study(study_name="prueba-1", storage=storage).trials
    # Los ensayos del hipercubo latino esperan con su número ya puesto hasta que se piden.
    assert [trial.state.name for trial in trials] == [
        "COMPLETE",
        "FAIL",
        "RUNNING",
        "RUNNING",
        *["WAITING"] * 4,
    ]
    document = search.report(study, declaration())
    assert (document["told"], document["pending"], document["failed"]) == (1, 2, 1)


def test_a_trial_that_never_reached_the_ledger_is_closed(tmp_path):
    import optuna

    root = repository(tmp_path)
    study = tmp_path / "study"
    search.ask(study, declaration(), root, 2)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(study / "optuna.journal"))
    )
    # Un corte justo después de pedir el ensayo a Optuna y antes de anotarlo.
    optuna.load_study(study_name="prueba-1", storage=storage).ask()
    batch = search.ask(study, declaration(), root, 1)
    assert [entry["number"] for entry in batch] == [3]
    trials = optuna.load_study(study_name="prueba-1", storage=storage).trials
    assert [trial.state.name for trial in trials] == [
        "RUNNING",
        "RUNNING",
        "FAIL",
        "RUNNING",
        *["WAITING"] * 4,
    ]


def test_a_changed_declaration_is_rejected(tmp_path):
    root = repository(tmp_path)
    study = tmp_path / "study"
    search.ask(study, declaration(), root, 2)
    with pytest.raises(ValueError, match="ha cambiado"):
        search.ask(study, declaration(seed=8), root, 2)
    # Pedir más confirmaciones no cambia la búsqueda.
    search.ask(study, declaration(confirm=5), root, 1)


@pytest.mark.parametrize(
    "changes, message",
    [
        (dict(final_test_access=True), "final_test_access"),
        (dict(objective=dict(split="evaluation")), "no se declara"),
        (dict(space=[dict(name="walk_forward.x", type="float", low=1, high=2)]), "recipe."),
        (dict(space=[dict(name="recipe.a", type="float", low=2, high=1)]), "low"),
        (dict(space=[dict(name="recipe.a", type="ordinal", values=[1])]), "values"),
        (dict(startup=1), "startup"),
        (dict(known=[dict(model="x", values={"recipe.learning_rate": 1.0})]), "known"),
    ],
)
def test_a_wrong_declaration_is_rejected(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        search.validate(declaration(**changes))


def test_a_field_that_the_base_recipe_lacks_is_a_mistake(tmp_path):
    root = repository(tmp_path)
    space = [dict(name="recipe.session_ic.weight", type="float", low=1.0, high=5.0)]
    with pytest.raises(ValueError, match="session_ic"):
        search.ask(tmp_path / "study", declaration(space=space, known=[]), root, 2)


def test_the_report_orders_by_validation_and_confirms_in_calibration(tmp_path):
    root = repository(tmp_path)
    study, state = tmp_path / "study", tmp_path / "state"
    batch = search.ask(study, declaration(), root, 8)
    run_batch(state, batch)
    write_run(state, "titans_mac", signal=0.3, noise_seed=500)
    search.tell(study, declaration(), state)
    document = search.report(study, declaration())
    order = [row["validation"]["mean"] for row in document["trials"]]
    assert order == sorted(order, reverse=True) and len(order) == 8
    assert document["told"] == 8 and document["pending"] == 0
    assert len(document["candidates"]) == 3
    assert [row["number"] for row in document["candidates"]] == [
        row["number"] for row in document["trials"][:3]
    ]
    for row in document["candidates"]:
        assert row["confirmed"] == (row["against_base"]["calibration"]["mean"] > 0)
    assert -1.0 <= document["concordance"] <= 1.0
    assert document["chance_maximum"] > 0
    assert document["final_test_opened"] is False
    text = search.markdown(document)
    assert "evalua" not in text.lower()
    assert f"| {document['trials'][0]['number']} |" in text


def test_the_command_asks_tells_and_reports(tmp_path, capsys):
    root = repository(tmp_path)
    study, state = tmp_path / "study", tmp_path / "state"
    declared = tmp_path / "search.json"
    declared.write_text(json.dumps(declaration()))
    common = ["--study", str(study), "--declaration", str(declared)]
    assert search.main(["ask", *common, "--repo", str(root), "--count", "8"]) == 0
    batch = json.loads((study / "trials.json").read_text())["trials"]
    run_batch(state, batch)
    assert search.main(["tell", *common, "--state", str(state)]) == 0
    output = tmp_path / "out"
    assert search.main(["report", *common, "--output", str(output)]) == 0
    capsys.readouterr()
    assert json.loads((output / "search.json").read_text())["kind"] == "campaign_search_report_v1"
    assert "| Ensayo |" in (output / "search.md").read_text(encoding="utf-8")
