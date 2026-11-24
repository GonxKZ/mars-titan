"""Ideas 4 y 7: los árbitros de herramientas, sus piezas y la orden sobre un estado sintético."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.stats import spearmanr

from mars_titan.evaluation import agent_core as ac
from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation import online_arbiters as arbiters
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots
from mars_titan.evaluation import tool_arbiters as ta
from tests.evaluation import test_agent_core as core

FOLDER = core.ROOT / "configs" / "experiments"
TODAY = FOLDER / "posttraining-tool-arbiters-hoy.json"
WP5 = FOLDER / "posttraining-tool-arbiters-wp5.json"
ACPTPA = FOLDER / "posttraining-tool-arbiters-acptpa.json"
SEEDS = (42, 43, 44)
STATE_TOOLS = ("titans_mac_nogain", "titans_mac", "mars_titan_m3", "mars_titan_cognitive")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_declarations_are_registered_and_differ_only_in_their_tools():
    today, digest = ta.checked_declaration(load(TODAY))
    wp5, other = ta.checked_declaration(load(WP5))
    assert {ta.REGISTERED[digest], ta.REGISTERED[other]} == {TODAY.name, WP5.name}
    kept = ("tools", "description")
    assert {k: v for k, v in today.items() if k not in kept} == {
        k: v for k, v in wp5.items() if k not in kept
    }
    organs = [
        f"titans_mac_nogain__{arm}" for arm in ("dense_bank", "holographic_delta", "asset_trace")
    ]
    assert wp5["tools"] == today["tools"] + [dict(name=n, source="organs") for n in organs]
    s2 = load(FOLDER / "posttraining-aggregator-s2.json")
    assert today["s2"] == {role: s2[role] for role in ("parent", "tree", "organ")}
    for field in ("min_assets", "splits", "selection_split", "align", "modes", "operable_scales"):
        assert today[field] == s2[field]
    assert today["rule"]["author"] == dict(
        mode="operable", vs_agent=0.0025, vs_uniform=0.003, min_t=2.0
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["rule"].update(min_t=1.5),
        lambda d: d["rule"]["author"].update(vs_agent=0.001),
        lambda d: d["arbiters"]["dodea"].update(ridges=[0.1]),
        lambda d: d["tools"].pop(),
        lambda d: d.update(seeds=[42]),
    ],
)
def test_any_change_of_the_rule_or_the_tools_is_refused(change):
    document = load(TODAY)
    change(document)
    with pytest.raises(ValueError, match="no es una de las registradas"):
        ta.checked_declaration(document)


def stream(sessions=12, size=3, horizon=1, seed=0):
    """Un flujo de mercado sintético como el de `online_aggregator._build_market_streams`."""
    rng = np.random.default_rng(seed)
    times = core.START + np.arange(sessions, dtype=np.int64) * core.DAY
    ics = rng.uniform(-0.2, 0.4, (sessions, size))
    covs = []
    for _ in range(sessions):
        z = rng.standard_normal((50, size))
        z[:, 1] = 0.9 * z[:, 0] + 0.1 * z[:, 1]
        z = (z - z.mean(axis=0)) / z.std(axis=0)
        covs.append(z.T @ z / len(z))
    return dict(
        times=times,
        matures=times + horizon * core.DAY,
        losses=(1.0 - ics) / 2.0,
        covs=np.asarray(covs),
        scored=np.ones(sessions, dtype=bool),
    )


def test_dodea_without_pending_sessions_and_redundancy_is_the_agent():
    flow = stream(horizon=1)
    flow["covs"] = np.repeat(np.eye(3)[None], len(flow["times"]), axis=0)
    real = ta.dodea_weights(flow, half_life=5.0, ridge=0.0)
    np.testing.assert_allclose(real, ta.agent_weights(flow), rtol=0, atol=1e-12)


def test_the_hint_adds_one_gap_per_pending_session_to_every_copy():
    state = arbiters.AdaNormalHedgeTV.start(np.ones(3))
    for losses in ([0.4, 0.5, 0.45], [0.42, 0.48, 0.5], [0.41, 0.52, 0.47]):
        state.update(np.asarray(losses), np.ones(3, dtype=bool))
    base = state.distribution(np.ones(3, dtype=bool))
    hint = np.array([0.40, 0.50, 0.46])
    gap = float(base @ hint) - hint
    regret = np.vstack([state.regret, np.zeros(3)])
    magnitude = np.vstack([state.magnitude, np.zeros(3)])
    expected = np.zeros(3)
    for row in range(regret.shape[0]):
        expected += (
            arbiters.weights(regret[row] + 2 * gap, magnitude[row] + 2 * np.abs(gap))
            / (row + 1) ** 2
        )
    expected /= expected.sum()
    np.testing.assert_allclose(ta.optimistic(state, hint, 2, base), expected, rtol=1e-13)
    # Con pistas iguales no hay optimismo; la mejor pista gana peso.
    np.testing.assert_allclose(ta.optimistic(state, np.full(3, 0.45), 3, base), base, rtol=1e-13)
    assert ta.optimistic(state, hint, 2, base)[0] > base[0]


def test_the_redundancy_discount_moves_weight_to_the_tool_that_differs():
    covariance = np.array([[1.0, 0.98, 0.1], [0.98, 1.0, 0.1], [0.1, 0.1, 1.0]])
    weights = ta.decorrelated(np.full(3, 1 / 3), covariance, 0.1)
    assert weights.sum() == pytest.approx(1.0) and np.all(weights >= 0)
    assert weights[2] > weights[0] == pytest.approx(weights[1])
    # Herramientas sin redundancia y sin regularización: la mezcla no cambia.
    mixture = np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(ta.decorrelated(mixture, np.eye(3), 0.0), mixture, atol=1e-15)


@pytest.mark.parametrize("size", [2, 3, 4, 6])
def test_the_placebo_permutation_has_no_fixed_points_and_is_reproducible(size):
    for seed in range(20):
        order = ta.derangement(size, np.random.default_rng(seed))
        assert sorted(order) == list(range(size)) and not np.any(order == np.arange(size))
        again = ta.derangement(size, np.random.default_rng(seed))
        np.testing.assert_array_equal(order, again)


def test_the_placebo_keeps_the_size_of_the_adjustments_and_breaks_their_assignment():
    flow = stream(sessions=30, horizon=3, seed=3)
    real = ta.dodea_weights(flow, half_life=5.0, ridge=0.3)
    fake = ta.dodea_weights(flow, half_life=5.0, ridge=0.3, placebo_seed=7)
    again = ta.dodea_weights(flow, half_life=5.0, ridge=0.3, placebo_seed=7)
    np.testing.assert_array_equal(fake, again)
    assert not np.allclose(real, fake)
    np.testing.assert_allclose(fake.sum(axis=1), 1.0)
    # Con herramientas idénticas la permutación no cambia nada.
    twin = stream(sessions=30, horizon=3, seed=3)
    twin["losses"] = np.repeat(twin["losses"][:, :1], 3, axis=1)
    twin["covs"] = np.ones_like(twin["covs"])
    np.testing.assert_allclose(
        ta.dodea_weights(twin, half_life=5.0, ridge=0.3),
        ta.dodea_weights(twin, half_life=5.0, ridge=0.3, placebo_seed=7),
        atol=1e-15,
    )


def test_no_weight_reads_a_loss_before_it_matures_or_a_later_redundancy():
    flow = stream(sessions=25, horizon=4, seed=5)
    before = ta.dodea_weights(flow, half_life=1.0, ridge=0.1)
    cut = 12
    changed = dict(flow, losses=flow["losses"].copy(), covs=flow["covs"].copy())
    # Pérdidas de sesiones que todavía no han madurado al decidir `cut`, y todo lo posterior.
    late = flow["matures"] > flow["times"][cut]
    changed["losses"][late] = np.random.default_rng(1).uniform(0, 1, changed["losses"][late].shape)
    changed["covs"][cut + 1 :] = np.eye(3)
    after = ta.dodea_weights(changed, half_life=1.0, ridge=0.1)
    np.testing.assert_array_equal(after[: cut + 1], before[: cut + 1])
    assert not np.array_equal(after[cut + 1 :], before[cut + 1 :])


def campaign(tmp_path, assets=core.ASSETS, quantiles=False):
    """Tres semillas: dos herramientas ven el mismo factor, otras ven otros y el árbol otro.

    Con `quantiles`, las herramientas del estado y el órgano llevan los cuantiles 0,1 y 0,9 (con
    otro generador, para que los datos de lo demás no cambien).
    """
    rng = np.random.default_rng(31)
    wide = np.random.default_rng(77)

    def bounds(prediction):
        if not quantiles:
            return None
        half = 0.6 + 0.4 * np.abs(wide.standard_normal(prediction.size))
        return dict(quantile_0100=prediction - half, quantile_0900=prediction + half)

    state, reference, organs = tmp_path / "estado", tmp_path / "arboles", tmp_path / "organos"
    known = dict(alpha={}, liquidity={}, volatility={}, matures={})
    shared, day = {}, 0
    for split, count in core.SESSIONS.items():
        rows = {name: [] for name in ("sample_id", "asset_id", "prediction_at", "target")}
        factors = []
        for session in range(count):
            instant = core.START + (day + session) * core.DAY
            z = rng.standard_normal((4, assets))
            rows["sample_id"] += [f"{split[0]}{session:03d}-{a:02d}" for a in range(assets)]
            rows["asset_id"] += [f"US/A{a:02d}" for a in range(assets)]
            rows["prediction_at"] += [instant] * assets
            rows["target"] += list(0.3 * z[0] + 0.2 * z[1] + 0.3 * z[2] + z[3])
            factors.append(z[:3])
        day += count + 2
        shared[split] = (rows, np.concatenate(factors, axis=1))
        ids, size = rows["sample_id"], len(rows["sample_id"])
        known["alpha"][split] = dict(zip(ids, 0.002 * rng.standard_normal(size), strict=True))
        known["liquidity"][split] = dict(zip(ids, rng.normal(15, 2, size), strict=True))
        known["volatility"][split] = dict(zip(ids, rng.normal(3, 0.5, size), strict=True))
        known["matures"][split] = dict(
            zip(ids, np.asarray(rows["prediction_at"], dtype=np.int64) + 3 * core.DAY, strict=True)
        )
    sees = dict(titans_mac_nogain=0, titans_mac=0, mars_titan_m3=1, mars_titan_cognitive=1)

    def write_run(folder, record, splits, values, intervals=True):
        for split in splits:
            rows, _ = shared[split]
            prediction = values(split)
            core._write(folder, split, rows, prediction, bounds(prediction) if intervals else None)
        (folder / "run.json").write_text(json.dumps(record))

    for seed in SEEDS:
        for model in STATE_TOOLS:
            write_run(
                state / "runs" / model / str(core.YEAR) / f"seed-{seed}",
                dict(
                    status="completed",
                    final_test_opened=False,
                    run_id=f"{model}-{seed}",
                    request=dict(view_sha256="f" * 64),
                ),
                og.SPLITS,
                lambda s, m=model: (
                    shared[s][1][sees[m]] + 0.8 * rng.standard_normal(shared[s][1].shape[1])
                ),
            )
        parent = state / "runs" / "titans_mac_nogain" / str(core.YEAR) / f"seed-{seed}"
        organ = organs / "runs" / "titans_mac_nogain__combined" / str(core.YEAR) / f"seed-{seed}"
        for split in og.AGGREGATED:
            rows, z = shared[split]
            own = pq.read_table(parent / f"{split}-predictions.parquet").column("prediction")
            prediction = own.to_numpy() + 0.5 * z[2]
            core._write(organ, split, rows, prediction, bounds(prediction))
        record = dict(
            status="completed",
            final_test_opened=False,
            model="titans_mac_nogain__combined",
            origin=core.YEAR,
            seed=seed,
            arm="combined",
            parent=dict(model="titans_mac_nogain", run_id=f"titans_mac_nogain-{seed}"),
            inputs=dict(
                emissions={
                    s: pilots._sha256(parent / f"{s}-predictions.parquet") for s in og.SPLITS
                }
            ),
            outputs={s: pilots._sha256(organ / f"{s}-predictions.parquet") for s in og.AGGREGATED},
        )
        (organ / "run.json").write_text(json.dumps(record))
    write_run(
        reference / "runs" / "lightgbm_technical" / str(core.YEAR) / "seed-42",
        # Como el registro real de los árboles (`dgx_tabular_reference`): sin `status` ni
        # `final_test_opened`, con su modelo, su semilla y `final_test_access`.
        dict(
            kind="dgx_tabular_reference",
            model="lightgbm_technical",
            year=core.YEAR,
            seed=42,
            final_test_access=False,
        ),
        og.SPLITS,
        lambda s: shared[s][1][2] + 0.8 * rng.standard_normal(shared[s][1].shape[1]),
        intervals=False,
    )
    manifest = state / "cohorts" / str(core.YEAR) / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    return dict(state=state, reference=reference, organs=organs, known=known)


def _arguments(paths, output, declaration=TODAY):
    arguments = ["medir", "--declaration", str(declaration), "--state", str(paths["state"])]
    arguments += ["--reference", str(paths["reference"]), "--origin", str(core.YEAR)]
    arguments += ["--output", str(output)]
    for seed in SEEDS:
        arguments += ["--organs", f"{seed}={paths['organs']}"]
    return arguments


def test_the_order_measures_an_origin_on_a_synthetic_state(tmp_path, monkeypatch, capsys):
    paths = campaign(tmp_path)
    monkeypatch.setattr(ta.ac, "known_rows", core._known(paths))
    output = tmp_path / "arbitros"
    assert ta.main(_arguments(paths, output)) == 0
    folder = output / str(core.YEAR)
    summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and summary["declaration"] == TODAY.name
    variants = {"s2", "agent", "agent_uniform", "dodea", "dodea_placebo", "tools_ridge"}
    assert variants <= set(summary["levels"]["operable"])
    assert set(summary["contrasts"]["label"]) == {name for name, _, _ in ta.CONTRASTS}
    assert set(summary["verdict"]) == {"enters", "reasons", "author_refuted", "reading"}
    grid = load(TODAY)["arbiters"]["dodea"]
    for by_seed in summary["chosen"].values():
        for chosen in by_seed.values():
            assert chosen["dodea"]["half_life"] in grid["hint_half_lives"]
            assert chosen["dodea_placebo"]["ridge"] in grid["ridges"]
    # El IC de una herramienta en la etiqueta, contado aparte con la correlación de Spearman.
    table = pq.read_table(
        paths["state"]
        / "runs"
        / "titans_mac_nogain"
        / str(core.YEAR)
        / "seed-42"
        / "evaluation-predictions.parquet"
    )
    instants = table.column("prediction_at").cast("int64").to_numpy()
    prediction, target = table.column("prediction").to_numpy(), table.column("target").to_numpy()
    expected = np.mean(
        [
            spearmanr(prediction[instants == i], target[instants == i])[0]
            for i in np.unique(instants)
        ]
    )
    measured = summary["levels"]["label"]["tool:titans_mac_nogain"]["seeds"]["42"]["mean"]
    assert measured == pytest.approx(expected, abs=1e-6)
    sessions = pq.read_table(folder / "sesiones.parquet")
    assert set(sessions.column("series").to_pylist()) >= variants
    # S2 es ridge_cov_active sobre el padre, el árbol y la señal del órgano, contado aparte.
    tables = {
        name: {s: og._table(path, s) for s in og.AGGREGATED}
        for name, path in (
            ("parent", paths["state"] / "runs" / "titans_mac_nogain" / str(core.YEAR) / "seed-42"),
            (
                "tree",
                paths["reference"] / "runs" / "lightgbm_technical" / str(core.YEAR) / "seed-42",
            ),
            (
                "organ",
                paths["organs"]
                / "runs"
                / "titans_mac_nogain__combined"
                / str(core.YEAR)
                / "seed-42",
            ),
        )
    }
    alpha, _, _, matures = core._known(paths)(None, None, tables["parent"], None)

    def own(name, split):
        return tables[name][split].column("prediction").to_numpy().astype(np.float64)

    members = {}
    for name, values in (
        ("titans_mac_nogain", {s: own("parent", s) for s in og.AGGREGATED}),
        ("lightgbm_technical", {s: own("tree", s) for s in og.AGGREGATED}),
        (og.SIGNAL, {s: own("organ", s) - own("parent", s) for s in og.AGGREGATED}),
    ):
        members[name] = og._write_component(
            tmp_path / "s2" / name, tables["parent"], values, alpha, matures
        )
    run = oa.run_aggregation(
        members,
        splits=og.AGGREGATED,
        selection_split="calibration",
        min_assets=30,
        align="strict",
        operable=False,
        alpha_col="alpha",
        maturity_col="matures_at",
        placebo_seed=20261122,
        operable_scales=tuple(load(TODAY)["operable_scales"]),
    )
    ics = oa.session_rank_ic(
        run["aligned"]["evaluation"],
        run["aggregators"]["ridge_cov_active"]["split_predictions"]["evaluation"],
        min_assets=30,
    )
    s2 = summary["levels"]["label"]["s2"]["seeds"]["42"]["mean"]
    assert s2 == pytest.approx(np.mean(list(ics.values())), abs=1e-12)
    assert "Lectura de la regla" in (folder / "resumen.md").read_text(encoding="utf-8")
    capsys.readouterr()
    assert ta.main(_arguments(paths, output)) == 0
    assert "ya calculado" in capsys.readouterr().out


def test_an_organ_of_another_fit_and_a_missing_input_fail_with_a_reason(
    tmp_path, monkeypatch, capsys
):
    paths = campaign(tmp_path)
    monkeypatch.setattr(ta.ac, "known_rows", core._known(paths))
    record = (
        paths["organs"]
        / "runs"
        / "titans_mac_nogain__combined"
        / str(core.YEAR)
        / "seed-43"
        / "run.json"
    )
    changed = json.loads(record.read_text())
    changed["parent"]["run_id"] = "otro-ajuste"
    record.write_text(json.dumps(changed))
    assert ta.main(_arguments(paths, tmp_path / "a")) == 1
    assert "no es el brazo titans_mac_nogain__combined" in capsys.readouterr().err
    missing = paths["state"] / "runs" / "mars_titan_m3" / str(core.YEAR) / "seed-44"
    (missing / "evaluation-predictions.parquet").unlink()
    assert ta.main(_arguments(paths, tmp_path / "b")) == og.MISSING_EXIT
    assert "mars_titan_m3" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (dict(final_test_access=True), "no son los árboles declarados"),
        (dict(model="xgboost_technical"), "no son los árboles declarados"),
        (dict(seed=43), "no son los árboles declarados"),
    ],
)
def test_a_tree_record_that_is_not_the_declared_one_is_refused(
    tmp_path, monkeypatch, capsys, change, reason
):
    paths = campaign(tmp_path)
    monkeypatch.setattr(ta.ac, "known_rows", core._known(paths))
    record = (
        paths["reference"] / "runs" / "lightgbm_technical" / str(core.YEAR) / "seed-42" / "run.json"
    )
    record.write_text(json.dumps(dict(json.loads(record.read_text()), **change)))
    assert ta.main(_arguments(paths, tmp_path / "x")) == 1
    assert reason in capsys.readouterr().err


def test_a_missing_tree_record_is_listed_and_nothing_is_written(tmp_path, monkeypatch, capsys):
    paths = campaign(tmp_path)
    monkeypatch.setattr(ta.ac, "known_rows", core._known(paths))
    record = (
        paths["reference"] / "runs" / "lightgbm_technical" / str(core.YEAR) / "seed-42" / "run.json"
    )
    record.unlink()
    # Una entrada que falta es un fallo temporal (75): la orden puede esperar a que llegue.
    assert ta.main(_arguments(paths, tmp_path / "x")) == 75
    assert "lightgbm_technical" in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_the_streams_give_dodea_the_covariance_of_the_session():
    # La Σ de DODEA es la del flujo del agregador: la covarianza de los rangos tipificados.
    rng = np.random.default_rng(2)
    p = rng.standard_normal((40, 3))
    z = np.column_stack([oa.standardized_ranks(p[:, k]) for k in range(3)]).astype(np.float64)
    assert np.allclose(np.diag(z.T @ z / 40), 1.0, atol=0.05)


def _contrast(mean, t):
    return dict(mean=mean, t_nw=t)


RULE = load(TODAY)["rule"]


@pytest.mark.parametrize(
    ("s2", "placebo", "agent", "uniform", "enters", "refuted"),
    [
        ((0.003, 2.5), (0.002, 2.1), (0.003, 2.2), (0.004, 2.4), True, False),
        ((0.003, 1.9), (0.002, 2.1), (0.003, 2.2), (0.004, 2.4), False, False),
        ((0.003, 2.5), (0.002, 1.5), (0.003, 2.2), (0.004, 2.4), False, False),
        ((0.003, 2.5), (0.002, 2.1), (0.0024, 3.0), (0.004, 2.4), True, True),
        ((0.003, 2.5), (0.002, 2.1), (0.003, 2.2), (0.004, 1.9), True, True),
        ((-0.001, 3.0), (0.002, 2.1), (0.003, 2.2), (0.004, 2.4), False, False),
    ],
)
def test_the_verdict_reads_the_declared_rule(s2, placebo, agent, uniform, enters, refuted):
    filler = _contrast(0.0, 0.0)
    main = {name: filler for name, _, _ in ta.CONTRASTS}
    main.update(
        dodea_vs_s2=_contrast(*s2),
        dodea_vs_placebo=_contrast(*placebo),
        dodea_vs_agent=_contrast(*agent),
        dodea_vs_uniform=_contrast(*uniform),
    )
    contrasts = dict(operable=main, label={name: filler for name, _, _ in ta.CONTRASTS})
    result = ta.verdict(contrasts, RULE)
    assert result["enters"] is enters and result["author_refuted"] is refuted


def test_the_seed_mean_uses_the_sessions_of_every_seed():
    left = {("US", 1): 0.1, ("US", 2): 0.3}
    right = {("US", 2): 0.5, ("US", 3): 0.0}
    assert ta.seed_mean([left, right]) == {("US", 2): pytest.approx(0.4)}


def aligned_panel(seed=0, sessions=(20, 25), assets=40, size=3):
    """Tramos alineados como los de `online_aggregator.run_aggregation`, con su madurez."""
    rng = np.random.default_rng(seed)
    aligned, day = {}, 0
    for split, count in zip(og.AGGREGATED, sessions, strict=True):
        rows = count * assets
        factor = rng.standard_normal(rows)
        predictions = np.column_stack(
            [factor + (0.5 + k) * rng.standard_normal(rows) for k in range(size)]
        ).astype(np.float32)
        instants = core.START + (day + np.repeat(np.arange(count), assets)) * core.DAY
        aligned[split] = dict(
            sessions=[
                ("US", int(core.START + (day + s) * core.DAY), s * assets, (s + 1) * assets)
                for s in range(count)
            ],
            target=(0.3 * factor + rng.standard_normal(rows)).astype(np.float32),
            predictions=predictions,
            model_names=[f"m{k}" for k in range(size)],
            matures_at=(instants + 3 * core.DAY).astype(np.int64),
        )
        day += count
    return aligned


def test_the_choice_of_dodea_uses_only_the_calibration():
    aligned = aligned_panel()
    grid = [(1.0, 0.1), (5.0, 1.0), (21.0, 3.0)]
    streams = oa._build_market_streams(aligned, og.AGGREGATED, min_assets=30)
    first = ta.chosen_dodea(aligned, streams, grid, min_assets=30)
    changed = {s: dict(d) for s, d in aligned.items()}
    rng = np.random.default_rng(9)
    changed["evaluation"]["target"] = rng.permutation(aligned["evaluation"]["target"])
    streams2 = oa._build_market_streams(changed, og.AGGREGATED, min_assets=30)
    second = ta.chosen_dodea(changed, streams2, grid, min_assets=30)
    assert first["choice"] == second["choice"] and first["scores"] == second["scores"]
    assert first["ics"]["calibration"] == second["ics"]["calibration"]
    assert first["choice"]["calibration_ic"] == pytest.approx(
        np.mean(list(first["ics"]["calibration"].values()))
    )
    assert first["choice"]["calibration_ic"] == max(first["scores"].values())


def test_dodea_matches_an_independent_count_on_a_small_flow():
    flow = stream(sessions=18, horizon=3, seed=11)
    flow["scored"][4] = False
    half_life, ridge = 2.0, 0.3
    decay = 0.5 ** (1 / half_life)
    times, matures = flow["times"], flow["matures"]
    expected = []
    for t in range(len(times)):
        done = [s for s in range(len(times)) if matures[s] <= times[t]]
        done.sort(key=lambda s: (matures[s], times[s]))
        state = arbiters.AdaNormalHedgeTV.start(np.ones(3))
        used = [s for s in done if flow["scored"][s]]
        for s in used:
            state.update(flow["losses"][s], np.ones(3, dtype=bool))
        base = state.distribution(np.ones(3, dtype=bool))
        pending = sum(1 for s in range(len(times)) if times[s] < times[t] < matures[s])
        if used and pending:
            powers = decay ** np.arange(len(used) - 1, -1, -1)
            hint = (powers[:, None] * flow["losses"][used]).sum(axis=0) / powers.sum()
            base = ta.optimistic(state, hint, pending, base)
        expected.append(ta.decorrelated(base, flow["covs"][t], ridge))
    np.testing.assert_allclose(
        ta.dodea_weights(flow, half_life=half_life, ridge=ridge), np.asarray(expected), rtol=1e-12
    )


# ---------------------------------------------------------------------------------------------
# Idea 7: ACPTPA


def test_the_acptpa_declaration_is_registered_and_uses_the_tools_of_today():
    document, digest = ta.checked_declaration(load(ACPTPA))
    assert ta.REGISTERED[digest] == ACPTPA.name
    kept = ("arbiters", "rule", "description")
    assert {k: v for k, v in document.items() if k not in kept} == {
        k: v for k, v in load(TODAY).items() if k not in kept
    }
    settings = document["arbiters"]["acptpa"]
    assert settings["point_tools"] == ["lightgbm_technical"]
    assert settings["quantiles"] == ["quantile_0100", "quantile_0900"] and settings["alpha"] == 0.2
    feedback = load(FOLDER / "posttraining-agent-v8-2013.json")["feedback"]
    assert settings["agent_abstention"] == dict(
        window=feedback["window"], margin=feedback["margin"]
    )
    assert document["rule"]["author"] == dict(
        mode="operable", liquid_reliability_vs_agent_abstaining=0.02, ic_vs_agent=0.002, min_t=2.0
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["arbiters"]["acptpa"].update(pid_gains=[1.0, 0.0, 0.0]),
        lambda d: d["arbiters"]["acptpa"].update(gamma_steps=[20.0]),
        lambda d: d["rule"]["author"].update(ic_vs_agent=0.001),
        lambda d: d["arbiters"]["acptpa"].update(point_tools=[]),
    ],
)
def test_any_change_of_acptpa_is_refused(change):
    document = load(ACPTPA)
    change(document)
    with pytest.raises(ValueError, match="no es una de las registradas"):
        ta.checked_declaration(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", 1.0),
        ("point_tools", [tool["name"] for tool in load(TODAY)["tools"]]),
        ("pid_gains", [1.0, -0.1, 0.1]),
        ("gamma_starts", [9.0]),
        ("agent_abstention", dict(window=0, margin=0.0)),
    ],
)
def test_the_acptpa_settings_say_what_is_wrong(field, value):
    document = load(ACPTPA)
    settings = dict(document["arbiters"]["acptpa"], **{field: value})
    names = [tool["name"] for tool in document["tools"]]
    with pytest.raises(ValueError, match="ACPTPA declara"):
        ta._checked_acptpa(settings, names, document["rule"])


def test_the_conformal_start_is_the_finite_sample_quantile():
    scores = np.arange(10.0)[:, None] * np.array([[1.0, -1.0]])
    np.testing.assert_array_equal(ta.conformal_start(scores, 0.2), [8.0, -1.0])
    np.testing.assert_array_equal(ta.conformal_start(np.array([[3.0], [1.0], [2.0]]), 0.2), [3.0])
    low, high = np.array([[0.0, -1.0]]), np.array([[1.0, 0.0]])
    np.testing.assert_array_equal(ta.interval_scores(low, high, np.array([0.25])), [[-0.25, 0.25]])


def pid_flow(sessions=14, rows=24, size=2, horizon=2, seed=0):
    rng = np.random.default_rng(seed)
    times = core.START + np.arange(sessions, dtype=np.int64) * core.DAY
    scores = [rng.standard_normal((rows, size)) * (1 + np.arange(size)) for _ in range(sessions)]
    groups = [np.where(np.arange(rows) % 3 == 2, -1, np.arange(rows) % 2) for _ in range(sessions)]
    return times, times + horizon * core.DAY, scores, groups


def test_the_pid_thresholds_match_an_independent_count():
    times, matures, scores, groups = pid_flow()
    alpha, gains = 0.2, (1.0, 0.1, 0.3)
    used = ta.pid_thresholds(times, matures, scores, groups, alpha=alpha, gains=gains, count=2)
    expected = np.full(used.shape, np.nan)
    state, applied = {}, set()
    for t in range(len(times)):
        ready = [s for s in range(len(times)) if matures[s] <= times[t] and s not in applied]
        for s in sorted(ready, key=lambda s: (matures[s], times[s])):
            applied.add(s)
            for g in range(2):
                for k in range(scores[s].shape[1]):
                    values = scores[s][groups[s] == g, k]
                    if (k, g) not in state:
                        rank = min(len(values), int(np.ceil((len(values) + 1) * (1 - alpha))))
                        start = np.sort(values)[rank - 1]
                        state[(k, g)] = dict(
                            start=start,
                            scale=np.mean(np.abs(values)),
                            q=start,
                            total=0.0,
                            last=None,
                        )
                        continue
                    if np.isnan(expected[s, k, g]):
                        continue
                    own = state[(k, g)]
                    error = np.mean(values > expected[s, k, g]) - alpha
                    own["total"] += error
                    change = 0.0 if own["last"] is None else error - own["last"]
                    own["last"] = error
                    own["q"] = own["start"] + own["scale"] * (
                        gains[0] * error + gains[1] * own["total"] + gains[2] * change
                    )
        for (k, g), own in state.items():
            expected[t, k, g] = own["q"]
    np.testing.assert_allclose(used, expected, rtol=1e-12, equal_nan=True)
    assert np.isnan(used[:2]).all() and np.isfinite(used[-1]).all()


def test_the_pid_brings_the_coverage_to_its_nominal_level():
    rng = np.random.default_rng(4)
    sessions, rows = 400, 300
    times = core.START + np.arange(sessions, dtype=np.int64) * core.DAY
    scores = [rng.standard_normal((rows, 1)) + (0.5 if s >= 200 else 0.0) for s in range(sessions)]
    groups = [np.zeros(rows, dtype=np.int64)] * sessions
    used = ta.pid_thresholds(
        times, times + core.DAY, scores, groups, alpha=0.2, gains=(1.0, 0.1, 0.1), count=1
    )
    coverage = np.array([np.mean(scores[s][:, 0] <= used[s, 0, 0]) for s in range(sessions)])
    assert abs(coverage[100:200].mean() - 0.8) < 0.01
    assert abs(coverage[300:].mean() - 0.8) < 0.01


def test_no_threshold_reads_a_score_before_it_matures():
    times, matures, scores, groups = pid_flow(sessions=20, horizon=3, seed=2)
    settings = dict(alpha=0.2, gains=(1.0, 0.1, 0.1), count=2)
    before = ta.pid_thresholds(times, matures, scores, groups, **settings)
    cut = 10
    late = [s + 5.0 if matures[i] > times[cut] else s for i, s in enumerate(scores)]
    after = ta.pid_thresholds(times, matures, late, groups, **settings)
    np.testing.assert_array_equal(after[: cut + 1], before[: cut + 1])
    assert not np.allclose(after[cut + 1 :], before[cut + 1 :], equal_nan=True)


def test_the_precision_favours_the_narrower_tool_of_each_asset():
    low = np.zeros((4, 2))
    high = np.array([[1.0, 3.0], [3.0, 1.0], [1.0, 1.0], [1.0, 3.0]])
    thresholds = np.zeros((2, 2))
    thresholds[:, 1] = np.nan
    groups = np.array([0, 0, 0, 1])
    omega = ta.precision(low, high, thresholds, groups, epsilon=0.05, gamma=2.0)
    np.testing.assert_allclose(omega[0], (np.array([1.0, 3.0]) + 0.05 * 2.0) ** -2.0)
    assert omega[1, 1] > omega[1, 0] and omega[2, 0] == omega[2, 1]
    np.testing.assert_array_equal(omega[3], [1.0, 1.0])
    flat = ta.precision(low, high, thresholds, groups, epsilon=0.05, gamma=0.0)
    np.testing.assert_array_equal(flat, np.ones((4, 2)))
    # Un umbral negativo recorta la anchura en cero; sin anchura en la fila, ponderación constante.
    shrunk = ta.precision(
        low, high, np.full((2, 1), -1.0), np.zeros(4, dtype=np.int64), epsilon=0.05, gamma=1.0
    )
    np.testing.assert_allclose(shrunk[0], [1 / 0.025, 1 / 1.025])
    np.testing.assert_array_equal(shrunk[2], [1.0, 1.0])


def acptpa_flow(sessions=30, rows=120, size=2, horizon=2, seed=0, informative=0):
    """Un flujo de mercado sintético: la herramienta `informative` ve el objetivo."""
    rng = np.random.default_rng(seed)
    times = core.START + np.arange(sessions, dtype=np.int64) * core.DAY
    z_sessions, y_sessions, parts = [], [], []
    for _ in range(sessions):
        target = rng.standard_normal(rows)
        z = rng.standard_normal((rows, size))
        z[:, informative] = target + 0.5 * rng.standard_normal(rows)
        z = (z - z.mean(axis=0)) / z.std(axis=0)
        liquid = np.zeros(rows, dtype=bool)
        liquid[: rows // 3] = True
        parts.append(
            dict(
                low=np.zeros((rows, size)),
                high=np.ones((rows, size)),
                groups=np.zeros(rows, dtype=np.int64),
                liquid=liquid,
            )
        )
        z_sessions.append(z.astype(np.float32))
        y_sessions.append(target.astype(np.float32))
    stream = dict(
        times=times,
        matures=times + horizon * core.DAY,
        z_sessions=z_sessions,
        y_sessions=y_sessions,
    )
    return stream, parts, np.zeros((sessions, size, 1)), np.full((sessions, size), 1.0 / size)


SETTINGS = dict(gamma_max=8.0, epsilon=0.05, min_assets=30)


def test_acptpa_without_gamma_is_the_constant_weighting():
    stream, parts, thresholds, _ = acptpa_flow(seed=1)
    rng = np.random.default_rng(2)
    for part in parts:
        part["high"] = rng.uniform(0.5, 2.0, part["high"].shape)
    base = rng.dirichlet(np.ones(2), size=len(parts))
    outputs, gammas = ta.acptpa_emission(
        stream, parts, base, thresholds, gamma_start=0.0, gamma_step=0.0, **SETTINGS
    )
    for session, output in enumerate(outputs):
        expected = stream["z_sessions"][session].astype(np.float64) @ base[session]
        np.testing.assert_allclose(output, expected, rtol=0, atol=1e-12)
    assert np.all(gammas == 0.0)


@pytest.mark.parametrize(("narrow", "rises"), [(0, True), (1, False)])
def test_the_cdt_raises_gamma_only_while_precision_helps_the_liquid_third(narrow, rises):
    stream, parts, thresholds, base = acptpa_flow(sessions=60, seed=3, informative=0)
    for part in parts:
        part["high"][:, 1 - narrow] = 3.0
    _, gammas = ta.acptpa_emission(
        stream, parts, base, thresholds, gamma_start=1.0, gamma_step=20.0, **SETTINGS
    )
    assert gammas[0] == gammas[1] == 1.0
    assert (gammas[-1] > 4.0) if rises else (gammas[-1] == 0.0)


def test_the_cdt_reads_only_the_liquid_third():
    stream, parts, thresholds, base = acptpa_flow(sessions=60, seed=5, informative=0)
    for session, part in enumerate(parts):
        z = stream["z_sessions"][session]
        z[~part["liquid"], 0] = -z[~part["liquid"], 0]
        part["high"][:, 1] = 3.0
    _, gammas = ta.acptpa_emission(
        stream, parts, base, thresholds, gamma_start=1.0, gamma_step=20.0, **SETTINGS
    )
    assert gammas[-1] > 4.0


def test_the_acptpa_placebo_moves_the_precision_to_other_assets():
    stream, parts, thresholds, base = acptpa_flow(sessions=20, seed=7)
    rng = np.random.default_rng(8)
    for part in parts:
        part["high"] = rng.uniform(0.5, 2.0, part["high"].shape)
    fixed = dict(gamma_start=2.0, gamma_step=0.0, **SETTINGS)
    real, _ = ta.acptpa_emission(stream, parts, base, thresholds, **fixed)
    fake, _ = ta.acptpa_emission(stream, parts, base, thresholds, placebo_seed=3, **fixed)
    again, _ = ta.acptpa_emission(stream, parts, base, thresholds, placebo_seed=3, **fixed)
    for left, right in zip(fake, again, strict=True):
        np.testing.assert_array_equal(left, right)
    session = 5
    assert not np.allclose(real[session], fake[session])
    part = parts[session]
    omega = ta.precision(
        part["low"], part["high"], thresholds[session], part["groups"], epsilon=0.05, gamma=2.0
    )
    seed = np.random.SeedSequence([3, ta.market_seed("US"), int(stream["times"][session])])
    order = ta.derangement(len(omega), np.random.default_rng(seed))
    expected = ta.mixed(stream["z_sessions"][session], base[session], omega[order])
    np.testing.assert_allclose(fake[session], expected, rtol=1e-13)
    # Con las mismas anchuras en todas las filas, el placebo no cambia nada.
    for part in parts:
        part["high"] = np.ones_like(part["high"]) * np.array([1.0, 2.0])
    real, _ = ta.acptpa_emission(stream, parts, base, thresholds, **fixed)
    fake, _ = ta.acptpa_emission(stream, parts, base, thresholds, placebo_seed=3, **fixed)
    for left, right in zip(real, fake, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-15)


def test_no_acptpa_emission_reads_a_label_before_it_matures():
    # La herramienta informativa es la estrecha: γ sube poco a poco y no llega a ninguno de sus
    # dos topes, que lo dejarían quieto (en 0 la emisión es la segura y γ ya no se mueve).
    stream, parts, thresholds, base = acptpa_flow(sessions=30, horizon=3, seed=9, informative=0)
    rng = np.random.default_rng(10)
    for part in parts:
        part["high"][:, 1] = 3.0
    moving = dict(gamma_start=1.0, gamma_step=1.0, **SETTINGS)
    before, early = ta.acptpa_emission(stream, parts, base, thresholds, **moving)
    cut = 15
    assert np.all((early[cut:] > 0.0) & (early[cut:] < SETTINGS["gamma_max"]))
    assert len(np.unique(early)) > 10
    late = stream["matures"] > stream["times"][cut]
    changed = dict(
        stream,
        y_sessions=[
            rng.permutation(y) if late[s] else y for s, y in enumerate(stream["y_sessions"])
        ],
    )
    after, later = ta.acptpa_emission(changed, parts, base, thresholds, **moving)
    for session in range(cut + 1):
        np.testing.assert_array_equal(after[session], before[session])
    np.testing.assert_array_equal(later[: cut + 1], early[: cut + 1])
    assert not np.array_equal(later[cut + 1 :], early[cut + 1 :])


def test_the_abstaining_agent_follows_the_rule_of_the_cognitive_agent():
    rng = np.random.default_rng(12)
    sessions = 15
    times = core.START + np.arange(sessions, dtype=np.int64) * core.DAY
    ics = rng.uniform(-0.1, 0.1, (sessions, 2))
    ics[5:9] = -0.05
    scored = np.ones(sessions, dtype=bool)
    scored[3] = False
    flow = dict(times=times, matures=times + 2 * core.DAY, ics=ics, scored=scored)
    exposure = ta.abstaining_exposure(flow, window=3, margin=0.0)
    for t in range(sessions):
        seen = [s for s in range(sessions) if flow["matures"][s] <= times[t] and scored[s]]
        seen = sorted(seen, key=lambda s: (flow["matures"][s], times[s]))[-3:]
        acting = bool(seen) and bool(np.any(ics[seen].mean(axis=0) > 0.0))
        assert exposure[t] == float(acting)
    assert exposure[0] == 0.0 and exposure[:].min() == 0.0 and exposure.max() == 1.0


def test_the_mixture_of_each_row_is_a_convex_combination():
    rng = np.random.default_rng(21)
    rows, size = 50, 3
    base = rng.dirichlet(np.ones(size))
    omega = rng.uniform(0.2, 5.0, (rows, size))
    # Con todas las herramientas iguales en una fila, la mezcla devuelve ese valor.
    common = rng.standard_normal(rows)
    z = np.repeat(common[:, None], size, axis=1)
    np.testing.assert_allclose(ta.mixed(z, base, omega), common, rtol=0, atol=1e-12)
    # Con herramientas distintas queda entre la menor y la mayor de la fila.
    z = rng.standard_normal((rows, size))
    output = ta.mixed(z, base, omega)
    assert np.all(output >= z.min(axis=1) - 1e-12)
    assert np.all(output <= z.max(axis=1) + 1e-12)
    expected = (base * omega * z).sum(axis=1) / (base * omega).sum(axis=1)
    np.testing.assert_allclose(output, expected, rtol=1e-13)


def test_a_reliability_equal_to_the_margin_does_not_end_the_abstention():
    sessions = 8
    times = core.START + np.arange(sessions, dtype=np.int64) * core.DAY
    flow = dict(
        times=times,
        matures=times + core.DAY,
        ics=np.zeros((sessions, 2)),
        scored=np.ones(sessions, dtype=bool),
    )
    assert np.all(ta.abstaining_exposure(flow, window=3, margin=0.0) == 0.0)
    flow["ics"][:, 1] = 0.25
    assert np.all(ta.abstaining_exposure(flow, window=3, margin=0.25) == 0.0)
    assert ta.abstaining_exposure(flow, window=3, margin=0.0)[-1] == 1.0


def test_the_liquid_ics_count_only_the_liquid_rows():
    rng = np.random.default_rng(13)
    sessions, rows = 4, 90
    target = rng.standard_normal(sessions * rows).astype(np.float32)
    prediction = (target + rng.standard_normal(sessions * rows)).astype(np.float32)
    liquid = rng.random(sessions * rows) < 0.5
    liquid[2 * rows : 3 * rows] = False
    liquid[2 * rows : 2 * rows + 10] = True
    data = dict(
        sessions=[("US", s, s * rows, (s + 1) * rows) for s in range(sessions)], target=target
    )
    abstained = np.array([False, True, False, False])
    result = ta.liquid_ics(data, prediction, liquid, min_assets=30, abstained=abstained)
    assert set(result) == {("US", 0), ("US", 1), ("US", 3)} and result[("US", 1)] == 0.0
    for s in (0, 3):
        part = slice(s * rows, (s + 1) * rows)
        mask = liquid[part]
        expected = spearmanr(prediction[part][mask], target[part][mask])[0]
        assert result[("US", s)] == pytest.approx(expected, abs=1e-6)
    assert ta.positive({"a": 0.1, "b": 0.0, "c": -0.2}) == {"a": 1.0, "b": 0.0, "c": 0.0}


@pytest.mark.parametrize(
    ("s2", "placebo", "reliability", "ic", "enters", "refuted"),
    [
        ((0.003, 2.5), (0.002, 2.1), (0.03, 2.5), (0.003, 2.4), True, False),
        ((0.003, 1.5), (0.002, 2.1), (0.03, 2.5), (0.003, 2.4), False, False),
        ((0.003, 2.5), (0.002, 2.1), (0.01, 2.5), (0.003, 2.4), True, False),
        ((0.003, 2.5), (0.002, 2.1), (0.03, 2.5), (0.001, 2.4), True, False),
        ((0.003, 2.5), (0.002, 2.1), (0.03, 1.9), (0.003, 1.9), True, True),
        ((0.003, 2.5), (-0.001, 3.0), (0.01, 3.0), (0.001, 3.0), False, True),
    ],
)
def test_the_acptpa_verdict_reads_the_declared_rule(s2, placebo, reliability, ic, enters, refuted):
    filler = _contrast(0.0, 0.0)
    main = {name: filler for name, _, _ in ta.ACPTPA_CONTRASTS}
    main.update(
        acptpa_vs_s2=_contrast(*s2),
        acptpa_vs_placebo=_contrast(*placebo),
        acptpa_vs_agent=_contrast(*ic),
    )
    contrasts = dict(operable=main, label={name: filler for name, _, _ in ta.ACPTPA_CONTRASTS})
    plain = {name: filler for name, _, _ in ta.LIQUID_CONTRASTS}
    liquid = {
        mode: dict(contrasts=dict(ic=dict(plain), reliability=dict(plain)))
        for mode in ("label", "operable")
    }
    liquid["operable"]["contrasts"]["reliability"]["acptpa_vs_abstaining"] = _contrast(*reliability)
    result = ta.acptpa_verdict(contrasts, liquid, load(ACPTPA)["rule"])
    assert result["enters"] is enters and result["author_refuted"] is refuted


def test_the_choice_of_acptpa_uses_only_the_calibration():
    aligned = aligned_panel(seed=4, sessions=(25, 25), assets=60)
    rng = np.random.default_rng(5)
    arrays = {}
    for split, data in aligned.items():
        rows = len(data["target"])
        center = data["predictions"].astype(np.float64)
        low = center - rng.uniform(0.5, 2.0, center.shape)
        high = center + rng.uniform(0.5, 2.0, center.shape)
        bounds = [(start, end) for _, _, start, end in data["sessions"]]
        arrays[split] = dict(
            low=low,
            high=high,
            scores=ta.interval_scores(low, high, data["target"]),
            groups=ac.within_session_groups(rng.standard_normal(rows), bounds, 2),
            liquid=ac.within_session_groups(rng.standard_normal(rows), bounds, 3) == 2,
        )
    settings = dict(
        load(ACPTPA)["arbiters"]["acptpa"],
        gamma_starts=[0.5, 2.0],
        gamma_steps=[5.0, 80.0],
        volatility_groups=2,
    )

    def choose(panel):
        streams = oa._build_market_streams(panel, og.AGGREGATED, min_assets=15)
        rows = {m: ta._stream_rows(st, panel, arrays) for m, st in streams.items()}
        thresholds = {
            m: ta.pid_thresholds(
                st["times"],
                st["matures"],
                [part["scores"] for part in rows[m]],
                [part["groups"] for part in rows[m]],
                alpha=0.2,
                gains=(1.0, 0.1, 0.1),
                count=2,
            )
            for m, st in streams.items()
        }
        base = {m: ta.agent_weights(st) for m, st in streams.items()}
        return ta.chosen_acptpa(panel, streams, rows, base, thresholds, settings, min_assets=15)

    first = choose(aligned)
    changed = {s: dict(d) for s, d in aligned.items()}
    changed["evaluation"]["target"] = np.random.default_rng(9).permutation(
        aligned["evaluation"]["target"]
    )
    second = choose(changed)
    assert first["choice"] == second["choice"] and first["scores"] == second["scores"]
    np.testing.assert_array_equal(
        first["predictions"]["calibration"], second["predictions"]["calibration"]
    )
    assert first["choice"]["calibration_ic"] == max(first["scores"].values())
    assert len(first["scores"]) == 4


def _s2_evaluation(tmp_path, paths, seed=42):
    """S2 en la etiqueta contado aparte: filas alineadas y predicción de evaluación."""
    year = str(core.YEAR)
    folders = dict(
        parent=paths["state"] / "runs" / "titans_mac_nogain" / year / f"seed-{seed}",
        tree=paths["reference"] / "runs" / "lightgbm_technical" / year / "seed-42",
        organ=paths["organs"] / "runs" / "titans_mac_nogain__combined" / year / f"seed-{seed}",
    )
    tables = {n: {s: og._table(p, s) for s in og.AGGREGATED} for n, p in folders.items()}
    alpha, _, _, matures = core._known(paths)(None, None, tables["parent"], None)

    def own(name, split):
        return tables[name][split].column("prediction").to_numpy().astype(np.float64)

    members = {}
    for name, values in (
        ("titans_mac_nogain", {s: own("parent", s) for s in og.AGGREGATED}),
        ("lightgbm_technical", {s: own("tree", s) for s in og.AGGREGATED}),
        (og.SIGNAL, {s: own("organ", s) - own("parent", s) for s in og.AGGREGATED}),
    ):
        members[name] = og._write_component(
            tmp_path / f"s2-{seed}" / name, tables["parent"], values, alpha, matures
        )
    run = oa.run_aggregation(
        members,
        splits=og.AGGREGATED,
        selection_split="calibration",
        min_assets=30,
        align="strict",
        operable=False,
        alpha_col="alpha",
        maturity_col="matures_at",
        placebo_seed=20261122,
        operable_scales=tuple(load(TODAY)["operable_scales"]),
    )
    return (
        run["aligned"]["evaluation"],
        run["aggregators"]["ridge_cov_active"]["split_predictions"]["evaluation"],
    )


def test_the_order_measures_acptpa_on_a_synthetic_state(tmp_path, monkeypatch):
    paths = campaign(tmp_path, assets=96, quantiles=True)
    monkeypatch.setattr(ta.ac, "known_rows", core._known(paths))
    output = tmp_path / "acptpa"
    assert ta.main(_arguments(paths, output, ACPTPA)) == 0
    folder = output / str(core.YEAR)
    summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
    assert summary["declaration"] == ACPTPA.name
    variants = {"s2", "agent", "agent_uniform", "agent_abstaining", "acptpa", "acptpa_placebo"}
    variants.add("tools_ridge")
    assert variants <= set(summary["levels"]["operable"])
    assert set(summary["contrasts"]["label"]) == {name for name, _, _ in ta.ACPTPA_CONTRASTS}
    assert set(summary["verdict"]) == {
        "enters",
        "reasons",
        "author_refuted",
        "author_holds",
        "reading",
    }
    assert set(summary["liquid"]["label"]["levels"]) == variants
    settings = load(ACPTPA)["arbiters"]["acptpa"]
    for by_seed in summary["chosen"].values():
        for chosen in by_seed.values():
            assert chosen["acptpa"]["gamma_start"] in settings["gamma_starts"]
            assert chosen["acptpa_placebo"]["gamma_step"] in settings["gamma_steps"]
            assert chosen["intervals"]["lightgbm_technical"] == "point"
            assert chosen["intervals"]["titans_mac"] == "quantiles"
            assert 0.6 < chosen["coverage"]["titans_mac"]["coverage"] < 0.95
    # El IC del tercio líquido de S2, contado aparte con la liquidez conocida de cada fila.
    data, prediction = _s2_evaluation(tmp_path, paths)
    liquidity = paths["known"]["liquidity"]["evaluation"]
    values = []
    for _, _, start, end in data["sessions"]:
        own = np.array([liquidity[s] for s in data["sample_id"][start:end]])
        third = np.argsort(np.argsort(own)) * 3 // len(own) == 2
        part = slice(start, end)
        values.append(spearmanr(prediction[part][third], data["target"][part][third])[0])
    measured = summary["liquid"]["label"]["levels"]["s2"]["seeds"]["42"]["ic"]
    assert measured == pytest.approx(np.mean(values), abs=1e-6)
    series = set(pq.read_table(folder / "sesiones.parquet").column("series").to_pylist())
    assert {"liquid:acptpa", "acptpa_placebo"} <= series
    text = (folder / "resumen.md").read_text(encoding="utf-8")
    assert "Tercio líquido" in text and "Cobertura del intervalo calibrado" in text


def test_a_tool_without_its_declared_interval_fails_with_a_reason(tmp_path, monkeypatch, capsys):
    paths = campaign(tmp_path, assets=96)
    monkeypatch.setattr(ta.ac, "known_rows", core._known(paths))
    assert ta.main(_arguments(paths, tmp_path / "x", ACPTPA)) == 1
    assert "no tiene intervalo" in capsys.readouterr().err
