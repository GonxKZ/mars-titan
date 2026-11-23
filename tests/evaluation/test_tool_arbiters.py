"""Ideas 4 y 7: los árbitros de herramientas, sus piezas y la orden sobre un estado sintético."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.stats import spearmanr

from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation import online_arbiters as arbiters
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots
from mars_titan.evaluation import tool_arbiters as ta
from tests.evaluation import test_agent_core as core

FOLDER = core.ROOT / "configs" / "experiments"
TODAY = FOLDER / "posttraining-tool-arbiters-hoy.json"
WP5 = FOLDER / "posttraining-tool-arbiters-wp5.json"
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


def campaign(tmp_path):
    """Tres semillas: dos herramientas ven el mismo factor, otras ven otros y el árbol otro."""
    rng = np.random.default_rng(31)
    state, reference, organs = tmp_path / "estado", tmp_path / "arboles", tmp_path / "organos"
    known = dict(alpha={}, liquidity={}, volatility={}, matures={})
    shared, day = {}, 0
    for split, count in core.SESSIONS.items():
        rows = {name: [] for name in ("sample_id", "asset_id", "prediction_at", "target")}
        factors = []
        for session in range(count):
            instant = core.START + (day + session) * core.DAY
            z = rng.standard_normal((4, core.ASSETS))
            rows["sample_id"] += [f"{split[0]}{session:03d}-{a:02d}" for a in range(core.ASSETS)]
            rows["asset_id"] += [f"US/A{a:02d}" for a in range(core.ASSETS)]
            rows["prediction_at"] += [instant] * core.ASSETS
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

    def write_run(folder, record, splits, values):
        for split in splits:
            rows, _ = shared[split]
            core._write(folder, split, rows, values(split))
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
            core._write(organ, split, rows, own.to_numpy() + 0.5 * z[2])
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
        dict(status="completed", final_test_opened=False, run_id="arbol"),
        og.SPLITS,
        lambda s: shared[s][1][2] + 0.8 * rng.standard_normal(shared[s][1].shape[1]),
    )
    manifest = state / "cohorts" / str(core.YEAR) / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    return dict(state=state, reference=reference, organs=organs, known=known)


def _arguments(paths, output):
    arguments = ["medir", "--declaration", str(TODAY), "--state", str(paths["state"])]
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
