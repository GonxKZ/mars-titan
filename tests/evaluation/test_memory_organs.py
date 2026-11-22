"""Órganos de memoria: emisión por fila de los dos brazos, su recibo y su lectura en la criba."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.evaluation import memory_organs as organs
from mars_titan.evaluation import posttraining_pilots as pilots
from mars_titan.memory import asset_residual_memory as arm
from mars_titan.memory import neighbour_residual_memory as nrm
from mars_titan.memory import view_keys as vk

HERE = Path(__file__).resolve()
ORIGIN, SEED = 2011, 42
SPLITS = pilots.SPLITS


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


views = _module("organ_views", HERE.parents[1] / "memory" / "test_view_keys.py")
report = _module("organ_screening", HERE.parents[2] / "scripts" / "dgx_screening_report.py")


def quiet(_):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(folder):
    """Cada archivo de una carpeta con su tamaño, su fecha y su huella."""
    return {
        str(path.relative_to(folder)): (path.stat().st_size, path.stat().st_mtime_ns, sha(path))
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def campaign(root, *, models=("a", "b"), assets=32, seed=0):
    """Un estado pequeño: la vista de su cohorte, las emisiones de cada modelo y su cola.

    Los modelos aciertan la mitad del objetivo y fallan un efecto persistente de cada activo,
    que la memoria por activo puede recoger.
    """
    state = root / "estado"
    manifest = views.write_view(state / "cohorts" / str(ORIGIN), assets=assets, seed=seed)
    document = json.loads(manifest.read_text())
    columns = {name: [] for name in ("asset", "at", "target", "matures", "part", "code")}
    for code, entry in enumerate(document["assets"]):
        labels = pq.read_table(
            Path(document["roots"]["labels"]) / entry["market"] / entry["symbol"] / "labels.parquet"
        )
        at = labels["prediction_at"].cast(pa.int64()).to_numpy()
        columns["asset"].append(np.full(len(at), f"{entry['market']}/{entry['symbol']}"))
        columns["at"].append(at)
        columns["target"].append(labels["target"].to_numpy())
        columns["matures"].append(np.r_[at[1:], at[-1] + pilots.DAY_US])
        columns["part"].append(np.array([p or "" for p in labels["partition"].to_pylist()]))
        columns["code"].append(np.full(len(at), code))
    data = {name: np.concatenate(values) for name, values in columns.items()}
    rng = np.random.default_rng(seed + 1)
    persistent = rng.normal(size=assets)
    rows = len(data["at"])
    base = 0.5 * data["target"] - 0.004 * persistent[data["code"]] + 0.01 * rng.normal(size=rows)
    predictions = {
        model: base / (1 + 2 * i) + 0.002 * rng.normal(size=rows) for i, model in enumerate(models)
    }
    maturity = {}
    for split in SPLITS:
        keep = data["part"] == split
        moments = pa.array(data["at"][keep], pa.int64()).cast(pa.timestamp("us", tz="UTC"))
        maturity[split] = pa.table(
            dict(
                asset_id=pa.array(data["asset"][keep]),
                market=pa.array(np.full(int(keep.sum()), "US")),
                prediction_at=moments,
                target=pa.array(data["target"][keep], pa.float64()),
                matures_at=pa.array(data["matures"][keep], pa.int64()),
            )
        )
        samples = [f"{a}/{t}" for a, t in zip(data["asset"][keep], data["at"][keep], strict=True)]
        for model in models:
            value = predictions[model][keep]
            table = pa.table(
                {
                    "sample_id": pa.array(samples),
                    "asset_id": maturity[split]["asset_id"],
                    "market": maturity[split]["market"],
                    "prediction_at": moments,
                    "target": maturity[split]["target"],
                    "prediction": pa.array(value, pa.float64()),
                    "zero": pa.array(np.zeros(len(value)), pa.float64()),
                    **{
                        column: pa.array(value + 0.002 * step, pa.float64())
                        for column, step in zip(
                            pilots.QUANTILE_COLUMNS, (-2, -1, 0, 1, 2), strict=True
                        )
                    },
                }
            )
            folder = state / "runs" / model / str(ORIGIN) / f"seed-{SEED}"
            folder.mkdir(parents=True, exist_ok=True)
            # La emisión llega desordenada, como las del modelo.
            shuffled = table.take(pa.array(rng.permutation(len(value))))
            pq.write_table(shuffled, folder / f"{split}-predictions.parquet")
    view = sha(manifest)
    (state / "queue").mkdir()
    for model in models:
        request = dict(view_sha256=view, window="fold-006", seed=SEED)
        folder = state / "runs" / model / str(ORIGIN) / f"seed-{SEED}"
        (folder / "run.json").write_text(
            json.dumps(dict(status="completed", run_id=model * 8, request=request))
        )
        record = dict(
            status="completed",
            model=model,
            tier="core",
            year=ORIGIN,
            seed=SEED,
            attempts=[dict(seconds=60.0)],
            result=dict(best_epoch=1, best_score=0.49),
        )
        (state / "queue" / f"{model}--{ORIGIN}--seed-{SEED}.json").write_text(json.dumps(record))
    return dict(state=state, manifest=manifest, view=view, maturity=maturity, models=models)


def provider(data):
    """La madurez de la vista sintética, en lugar de leerla con el lector de la campaña."""

    def maturity(manifest, view_sha256):
        assert Path(manifest) == data["manifest"] and view_sha256 == data["view"]
        return data["maturity"]

    return maturity


def declaration(**changes):
    document = {
        "schema_version": 1,
        "kind": organs.KIND,
        "final_test_access": False,
        "description": "prueba",
        "min_assets": 30,
        "models": ["a", "b"],
        "inputs": {
            "source": "view_memory_keys_v1",
            "columns": 92,
            "projection_dim": 8,
            "projection_seed": 53,
        },
        "asset": {
            "half_lives": [5, 21],
            "kappas": [0.05, 0.1, 0.2],
            "lab": {"half_life": 21, "kappa": 0.1},
        },
        "neighbours": {
            "window": 5,
            "k": 20,
            "tau": 0.2,
            "clip": 3.0,
            "kappas": [0.05, 0.1, 0.2, 0.4],
        },
        "combination": {"kappas": [0.0, 0.05, 0.1, 0.2]},
        "placebo_seed": 3,
    }
    document.update(changes)
    return document


def emit(data, keys, output, document=None, seeds=(SEED,)):
    return organs.emit(
        document or declaration(),
        data["state"],
        ORIGIN,
        list(seeds),
        keys,
        output,
        log=quiet,
        maturity=provider(data),
    )


def run(output, model, arm_name):
    return output / "runs" / f"{model}__{arm_name}" / str(ORIGIN) / f"seed-{SEED}"


def study(data, folder, document):
    """Las entradas del estudio de memorias leídas por su cuenta: tramos, predicciones y claves."""
    reference, predictions = None, {}
    for model in document["models"]:
        parent = data["state"] / "runs" / model / str(ORIGIN) / f"seed-{SEED}"
        splits = {
            name: pilots.load_split(
                pq.read_table(parent / f"{name}-predictions.parquet"), data["maturity"][name], name
            )
            for name in SPLITS
        }
        reference = reference or splits
        predictions[model] = np.concatenate([splits[name].prediction for name in SPLITS])
    assets = np.concatenate([reference[name].asset for name in SPLITS])
    days = np.concatenate([reference[name].prediction_at for name in SPLITS]) // pilots.DAY_US
    raw, _ = pilots.lab_inputs(folder, folder / "names.json", assets, days)
    keys = nrm.keys_from_inputs(raw, nrm.projection(92, 8, seed=53))
    return reference, predictions, keys


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    """Un estado, sus claves y la emisión de los órganos, para las pruebas que solo leen."""
    root = tmp_path_factory.mktemp("organos")
    data = campaign(root)
    keys = root / "claves"
    vk.build(data["manifest"], keys / data["view"], log=quiet)
    before = snapshot(data["state"])
    output = root / "salida"
    summary = emit(data, keys, output)
    return dict(data, keys=keys, output=output, summary=summary, before=before)


def test_each_arm_ranks_every_session_as_the_memory_study_measured_it(emitted):
    document = declaration()
    reference, predictions, keys = study(emitted, emitted["keys"] / emitted["view"], document)
    results, _ = pilots.memory_study(reference, predictions, keys, document, log=quiet)
    for model in ("a", "b"):
        # Los datos tienen un efecto por activo que el modelo no ve: el brazo principal suma.
        assert results[model]["evaluation"]["combined"] > results[model]["evaluation"]["parent"]
        for arm_name in organs.ARMS:
            for split in organs.EMITTED:
                data = report.load_split(run(emitted["output"], model, arm_name), split)
                values = list(report.session_ic(data).values())
                assert len(values) == len(reference[split].sessions)
                assert np.mean(values) == pytest.approx(results[model][split][arm_name], abs=1e-12)


def test_the_emission_keeps_the_rows_and_schema_and_moves_the_five_quantiles_with_the_median(
    emitted,
):
    for model in ("a", "b"):
        parent_run = emitted["state"] / "runs" / model / str(ORIGIN) / f"seed-{SEED}"
        for arm_name in organs.ARMS:
            folder = run(emitted["output"], model, arm_name)
            assert not (folder / "validation-predictions.parquet").exists()
            for split in organs.EMITTED:
                parent = pq.read_table(parent_run / f"{split}-predictions.parquet")
                child = pq.read_table(folder / f"{split}-predictions.parquet")
                assert child.schema == parent.schema and child.num_rows == parent.num_rows
                for column in (
                    "sample_id",
                    "asset_id",
                    "market",
                    "prediction_at",
                    "target",
                    "zero",
                ):
                    assert child[column].equals(parent[column])
                moved = child["prediction"].to_numpy() - parent["prediction"].to_numpy()
                assert np.abs(moved).max() > 0
                for column in pilots.QUANTILE_COLUMNS:
                    shift = child[column].to_numpy() - parent[column].to_numpy()
                    np.testing.assert_allclose(shift, moved, rtol=0, atol=1e-15)


def test_an_arm_reads_no_label_before_it_matures_and_is_chosen_in_the_calibration(emitted):
    document = declaration()
    reference, predictions, keys = study(emitted, emitted["keys"] / emitted["view"], document)
    _, _, first = pilots.memory_study(
        reference, predictions, keys, document, organs=True, log=quiet
    )
    evaluation = reference["evaluation"]
    _, _, start, end = evaluation.sessions[20]
    # Las etiquetas desde la sesión 20 de la evaluación cambian; maduran en la sesión siguiente.
    target = evaluation.target.copy()
    target[start:] = np.random.default_rng(9).normal(0, 0.01, len(target) - start)
    changed = dict(reference, evaluation=pilots.replace_target(evaluation, target))
    _, _, second = pilots.memory_study(changed, predictions, keys, document, organs=True, log=quiet)
    cut = reference["validation"].rows + reference["calibration"].rows + end
    for model in ("a", "b"):
        for arm_name in organs.ARMS:
            before = first["models"][model]["arms"][arm_name]
            after = second["models"][model]["arms"][arm_name]
            assert after["selection"] == before["selection"]
            np.testing.assert_array_equal(after["signal"][:cut], before["signal"][:cut])
            assert not np.array_equal(after["signal"][cut:], before["signal"][cut:])


def test_nothing_is_written_inside_the_campaign_state(emitted, tmp_path):
    assert snapshot(emitted["state"]) == emitted["before"]
    for keys, output in (
        (emitted["keys"], emitted["state"] / "organos"),
        (emitted["state"] / "claves", tmp_path / "salida"),
    ):
        with pytest.raises(ValueError, match="dentro del estado"):
            emit(emitted, keys, output)
    assert snapshot(emitted["state"]) == emitted["before"]


def test_the_receipt_names_the_choice_the_inputs_and_a_state_that_restores(emitted):
    document = declaration()
    reference, predictions, keys = study(emitted, emitted["keys"] / emitted["view"], document)
    results, _ = pilots.memory_study(reference, predictions, keys, document, log=quiet)
    keys_receipt = json.loads((emitted["keys"] / emitted["view"] / "keys.json").read_text())
    parent = emitted["state"] / "runs" / "a" / str(ORIGIN) / f"seed-{SEED}"
    receipts = {
        name: json.loads((run(emitted["output"], "a", name) / "run.json").read_text())
        for name in organs.ARMS
    }
    chosen = results["a"]["selection"]
    assert receipts["combined"]["selection"] == dict(
        selected_on="calibration",
        half_life=chosen["asset"]["half_life"],
        kappa_asset=chosen["combined"]["kappa_asset"],
        kappa_neighbours=chosen["combined"]["kappa_neighbours"],
    )
    assert receipts["asset_lab"]["selection"] == dict(selected_on="lab", half_life=21, kappa=0.1)
    joined = {
        name: np.concatenate([getattr(reference[split], field) for split in SPLITS])
        for name, field in (
            ("markets", "market"),
            ("assets", "asset"),
            ("prediction_at", "prediction_at"),
            ("matures_at", "matures_at"),
            ("target", "target"),
        )
    }
    for name, receipt in receipts.items():
        folder = run(emitted["output"], "a", name)
        assert receipt["status"] == "completed" and receipt["final_test_opened"] is False
        assert receipt["model"] == f"a__{name}" and receipt["parent"]["model"] == "a"
        assert receipt["request"]["view_sha256"] == emitted["view"]
        assert receipt["inputs"]["emissions"] == {
            split: sha(parent / f"{split}-predictions.parquet") for split in SPLITS
        }
        assert receipt["inputs"]["keys"]["view_sha256"] == emitted["view"]
        assert receipt["inputs"]["keys"]["files"] == keys_receipt["outputs"]
        assert receipt["outputs"] == {
            split: sha(folder / f"{split}-predictions.parquet") for split in organs.EMITTED
        }
        for split in organs.EMITTED:
            assert receipt["ic"][split] == pytest.approx(results["a"][split][name], abs=1e-12)
        saved = receipt["state"]["asset"]
        assert sha(emitted["output"] / saved["file"]) == saved["sha256"]
        restored = arm.AssetResidualMemory.restore(
            organs.load_state(emitted["output"] / saved["file"])
        )
        assert restored.state_sha256() == saved["state_sha256"]
        # Es el estado de la memoria del brazo al terminar la evaluación.
        half_life = float(receipt["selection"]["half_life"])
        direct = arm.AssetResidualMemory(arm.AssetMemoryConfig(half_life=half_life))
        arm.replay(direct, kappas=(0.0, 1.0), prediction=predictions["a"], **joined)
        assert direct.state_sha256() == saved["state_sha256"]
    assert "neighbours" not in receipts["asset_lab"]["state"]
    near = receipts["combined"]["state"]["neighbours"]
    assert near["models"] == ["a", "b"]
    restored = nrm.NeighbourMemory.restore(organs.load_state(emitted["output"] / near["file"]))
    assert restored.export()["state_sha256"] == near["state_sha256"]
    direct = nrm.NeighbourMemory(nrm.NeighbourConfig(window=5, k=20, tau=0.2, clip=3.0), models=2)
    nrm.replay(
        direct,
        markets=joined["markets"],
        prediction_at=joined["prediction_at"],
        matures_at=joined["matures_at"],
        keys=keys,
        predictions=np.column_stack([predictions["a"], predictions["b"]]),
        target=joined["target"],
    )
    assert direct.export()["state_sha256"] == near["state_sha256"]


def test_the_screening_report_reads_the_organ_root_as_one_more_state(emitted):
    controls = {f"{model}__{name}": model for model in ("a", "b") for name in organs.ARMS}
    document = report.build(
        emitted["state"],
        [ORIGIN],
        SEED,
        controls,
        {},
        references=[emitted["output"]],
        offset=True,
        decision_splits=("calibration",),
    )
    origin = document["origins"][0]
    assert sorted(origin["models"]) == [
        "a",
        "a__asset_lab",
        "a__combined",
        "b",
        "b__asset_lab",
        "b__combined",
    ]
    assert origin["missing_splits"] == []
    for name in controls:
        receipt = json.loads((run(emitted["output"], *name.split("__")) / "run.json").read_text())
        model = origin["models"][name]
        assert model["state"] == str(emitted["output"])
        assert model["ic"]["evaluation"]["mean"] == pytest.approx(
            receipt["ic"]["evaluation"], abs=1e-12
        )
        assert set(origin["offset"]["models"][name]["free"]) == set(organs.EMITTED)
    contrasts = {row["model"]: row for row in origin["contrasts"]}
    assert set(contrasts) == set(controls)
    assert {row["evaluation"]["sessions"] for row in contrasts.values()} == {50}


def test_the_keys_are_built_from_the_view_once_and_checked_against_the_runs(tmp_path):
    data = campaign(tmp_path, models=("a",))
    document = declaration(models=["a"])
    keys = tmp_path / "claves"
    emit(data, keys, tmp_path / "uno", document)
    folder = keys / data["view"]
    receipt = json.loads((folder / "keys.json").read_text())
    assert receipt["kind"] == vk.KIND and receipt["view_sha256"] == data["view"]
    stamp = (folder / "X.npy").stat().st_mtime_ns
    emit(data, keys, tmp_path / "dos", document)
    assert (folder / "X.npy").stat().st_mtime_ns == stamp
    # Unas claves de otra vista o retocadas no se usan.
    (folder / "keys.json").write_text(json.dumps(dict(receipt, view_sha256="0" * 64)))
    with pytest.raises(ValueError, match="otra vista"):
        emit(data, keys, tmp_path / "tres", document)
    (folder / "keys.json").write_text(json.dumps(receipt))
    np.save(folder / "X.npy", np.load(folder / "X.npy") + np.float32(1))
    with pytest.raises(ValueError, match="X.npy"):
        emit(data, keys, tmp_path / "cuatro", document)
    assert not (tmp_path / "tres" / "runs").exists() and not (tmp_path / "cuatro" / "runs").exists()


def test_a_model_emits_the_same_alone_or_with_others_and_a_missing_one_is_recorded(tmp_path):
    data = campaign(tmp_path)
    keys = tmp_path / "claves"
    vk.build(data["manifest"], keys / data["view"], log=quiet)
    together = emit(data, keys, tmp_path / "juntos", declaration(models=["a", "b", "c"]))
    (seed,) = together["runs"]
    assert seed["models"] == ["a", "b"] and seed["missing"] == ["c"] and not seed["skipped"]
    stem = f"{ORIGIN}--seed-{SEED}"
    written = json.loads((tmp_path / "juntos" / "studies" / f"{stem}.json").read_text())
    assert written["missing_models"] == ["c"] and sorted(written["models"]) == ["a", "b"]
    assert written["final_test_opened"] is False
    assert not (tmp_path / "juntos" / "runs" / "c__combined").exists()
    assert sorted(p.name for p in (tmp_path / "juntos" / "queue").iterdir()) == sorted(
        f"{m}__{name}--{stem}.json" for m in ("a", "b") for name in organs.ARMS
    )
    alone = emit(data, keys, tmp_path / "solo", declaration(models=["a"]))
    assert alone["runs"][0]["models"] == ["a"]
    for name in organs.ARMS:
        left, right = run(tmp_path / "juntos", "a", name), run(tmp_path / "solo", "a", name)
        receipts = [json.loads((folder / "run.json").read_text()) for folder in (left, right)]
        assert receipts[0]["selection"] == receipts[1]["selection"]
        for split in organs.EMITTED:
            assert receipts[0]["ic"][split] == pytest.approx(receipts[1]["ic"][split], abs=1e-12)
            file = f"{split}-predictions.parquet"
            together, single = pq.read_table(left / file), pq.read_table(right / file)
            if name == "asset_lab":
                assert together.equals(single)
            else:
                # La lectura por vecinos pondera los valores de todos los modelos a la vez y el
                # orden de la suma depende de cuántos son: solo cambia el redondeo.
                assert together.schema == single.schema
                np.testing.assert_allclose(
                    together["prediction"].to_numpy(), single["prediction"].to_numpy(), atol=1e-15
                )
    # Lo emitido no se rehace y una raíz no mezcla declaraciones.
    before = snapshot(tmp_path / "juntos")
    again = emit(data, keys, tmp_path / "juntos", declaration(models=["a", "b", "c"]))
    assert again["runs"][0]["skipped"] and snapshot(tmp_path / "juntos") == before
    with pytest.raises(ValueError, match="otra declaración"):
        emit(data, keys, tmp_path / "juntos", declaration(models=["a"]))
    with pytest.raises(ValueError, match="Ningún modelo"):
        emit(data, keys, tmp_path / "nada", declaration(models=["c"]))


def test_an_arm_that_would_move_a_constant_session_fails_with_its_coordinates():
    sessions = (("US", 5, 0, 3), ("US", 6, 3, 6))
    prediction = np.array([0.1, 0.2, 0.3, 0.5, 0.5, 0.5])
    quantiles = prediction[:, None] + 0.01 * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    dispersion = np.r_[np.full(3, prediction[:3].std()), np.zeros(3)]
    signal = np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.0])
    moved, moved_quantiles = organs.shifted(sessions, prediction, quantiles, dispersion, signal)
    np.testing.assert_array_equal(moved, prediction + dispersion * signal)
    np.testing.assert_array_equal(moved_quantiles, quantiles + (dispersion * signal)[:, None])
    with pytest.raises(ValueError, match="US 6"):
        organs.shifted(sessions, prediction, quantiles, dispersion, np.r_[signal[:3], 0.2, 0, 0])


@pytest.mark.parametrize(
    "changes, message",
    [
        (dict(final_test_access=True), "test final"),
        (dict(inputs=dict(declaration()["inputs"], source="tabular_lab_cache_v1")), "vista"),
        (dict(combination={"kappas": [0.05, 0.1]}), "cero"),
        (dict(asset=dict(declaration()["asset"], half_lives=[5])), "rejilla"),
        (dict(runs=[]), "campos"),
        (dict(models=[]), "modelos"),
    ],
)
def test_the_declaration_is_checked_before_reading_anything(changes, message):
    organs.checked_organs(declaration())
    shipped = HERE.parents[2] / "configs" / "experiments" / "posttraining-memory-organs-v1.json"
    organs.checked_organs(json.loads(shipped.read_text()))
    with pytest.raises(ValueError, match=message):
        organs.checked_organs(declaration(**changes))


def test_the_relay_line_starts_the_emission_in_its_own_unit_without_waiting(capsys):
    options = dict(
        label="v11",
        state=Path("/e/v11"),
        origin=2013,
        seeds=[42, 43],
        tree=Path("/t"),
        python=Path("/p/bin/python"),
        keys=Path("/k"),
        output=Path("/o"),
    )
    line = organs.relay_line(**options)
    kind, tool, arguments = line.split("|", 2)
    assert kind == "orden" and tool == "/usr/bin/systemd-run" and "|" not in arguments
    words = arguments.split()
    assert words[:4] == ["--user", "--collect", "--unit", "mars-titan-memorias-v11-2013"]
    for word in ("--slice=mars-titan.slice", "CPUWeight=10", "Nice=15", "MemoryMax=8G"):
        assert word in words
    assert "MemorySwapMax=0" in words and "--wait" not in words
    command = words[words.index("/p/bin/python") :]
    assert command[:3] == ["/p/bin/python", "-m", "mars_titan.evaluation.memory_organs"]
    args = organs.parser().parse_args(command[3:])
    assert args.command == "emitir" and args.state == Path("/e/v11") and args.origin == 2013
    assert args.seed == [42, 43] and args.keys == Path("/k") and args.output == Path("/o")
    assert args.declaration == Path("/t/configs/experiments/posttraining-memory-organs-v1.json")
    with pytest.raises(ValueError, match="espacio"):
        organs.relay_line(**dict(options, state=Path("/e/con espacio")))
    arguments = ["linea", "--label", "v11", "--state", "/e/v11", "--origin", "2013"]
    arguments += ["--seed", "42", "--seed", "43", "--tree", "/t", "--python", "/p/bin/python"]
    assert organs.main([*arguments, "--keys", "/k", "--output", "/o"]) == 0
    assert capsys.readouterr().out.strip() == line
