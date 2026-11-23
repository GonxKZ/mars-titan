"""F6: la declaración, las piezas puras y la orden de un origen sobre un estado sintético."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.stats import rankdata, spearmanr

from mars_titan.evaluation import holographic as holo
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots
from tests.evaluation import test_agent_core as core

DECLARATION = core.ROOT / "configs" / "experiments" / "posttraining-holographic.json"
SEEDS = (42, 43, 44)
STATIC = "titans_mac_disabled_nogain"
IGNORED = ["description", "models"]
WP5 = {
    arm: core.ROOT
    / "configs"
    / "experiments"
    / f"posttraining-holographic-wp5-{arm.replace('_', '-')}.json"
    for arm in holo.WP5_SHA256
}


def declaration():
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


def test_the_declaration_is_the_registered_one():
    document, digest = holo.checked_declaration(declaration())
    assert digest == holo.REGISTERED_SHA256
    assert document["rule"]["min_gain"] == 0.002 and document["static"] == STATIC
    assert document["seeds"] == list(SEEDS)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["rule"].update(min_gain=0.001),
        lambda d: d["rule"].update(min_t=1.5),
        lambda d: d["rule"].update(control_min_t=1.0),
        lambda d: d.update(seeds=[42, 43]),
        lambda d: d.update(core="titans_mac"),
    ],
)
def test_any_change_of_the_rule_is_refused(change):
    document = declaration()
    change(document)
    with pytest.raises(ValueError, match="no es la registrada para F6"):
        holo.checked_declaration(document)


def test_session_ranks_by_hand():
    ranks = holo.session_ranks(np.array([3.0, 1.0, 2.0, 5.0, 5.0]), [(0, 3), (3, 5)])
    assert ranks.tolist() == [1.0, 1 / 3, 2 / 3, 0.75, 0.75]


def _contrast(mean, t):
    return dict(sessions=100, mean=mean, t_nw=t, interval=None)


@pytest.mark.parametrize(
    ("label", "operable", "control", "enters", "organs"),
    [
        ((0.003, 2.5), (0.001, 1.0), (0.002, 2.1), True, True),
        ((0.003, 2.5), (0.001, 1.0), (0.002, 1.5), True, False),
        ((0.0019, 3.0), (0.001, 1.0), (0.002, 2.5), False, True),
        ((0.003, 1.9), (0.001, 1.0), (0.002, 2.5), False, True),
        ((0.003, 2.5), (-0.0001, 0.0), (0.002, 2.5), False, True),
    ],
)
def test_the_verdict_reads_the_agreed_rule(label, operable, control, enters, organs):
    contrasts = dict(
        label=dict(c3h3_vs_c3=_contrast(*label), c3h3_vs_c3e3=_contrast(*control)),
        operable=dict(c3h3_vs_c3=_contrast(*operable)),
    )
    result = holo.verdict(contrasts, declaration()["rule"])
    assert result["enters"] is enters and result["organs_matter"] is organs
    if not enters:
        assert result["reading"] == "H se queda como órgano del agregador"
    elif not organs:
        assert "segunda red" in result["reading"]


def test_paired_contrasts_use_the_common_sessions():
    tool = og._screening()
    left = {("US", t): 0.1 + 0.01 * t for t in range(30)}
    right = {("US", t): 0.1 for t in range(1, 30)}
    entry = holo.paired(tool, left, right)
    assert entry["sessions"] == 29
    assert entry["mean"] == pytest.approx(np.mean([0.01 * t for t in range(1, 30)]))
    assert entry["interval"][0] < entry["mean"] < entry["interval"][1]
    assert holo.ic_correlation(left, {k: 2 * v for k, v in left.items()}) == pytest.approx(1.0)


def test_organ_roots_must_share_the_declaration(tmp_path):
    base = dict(kind="k", asset=dict(a=1), models=[STATIC], description="uno")
    documents = dict(a=base, b=dict(base, models=["x", STATIC], description="dos"))
    documents["c"] = dict(base, asset=dict(a=2))
    for name, document in documents.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "declaration.json").write_text(json.dumps(document))
    roots = holo.organ_declarations(
        [tmp_path / "a", tmp_path / "b", tmp_path / "b"], STATIC, IGNORED
    )
    assert len(roots) == 2
    with pytest.raises(ValueError, match="no comparten declaración"):
        holo.organ_declarations([tmp_path / "a", tmp_path / "c"], STATIC, IGNORED)
    with pytest.raises(ValueError, match="no declara órganos"):
        holo.organ_declarations([tmp_path / "a"], "otro", IGNORED)


def campaign(tmp_path, *, organ_strength=0.6, core_noise=0.8, static_noise=0.8, arm="combined"):
    """Tres semillas de núcleo, secundario y estático que ven un factor; el órgano ve otro."""
    rng = np.random.default_rng(21)
    state = tmp_path / "estado"
    roots = {42: tmp_path / "organos-a", 43: tmp_path / "organos-b", 44: tmp_path / "organos-b"}
    for root in set(roots.values()):
        root.mkdir(parents=True, exist_ok=True)
        document = dict(kind="organos", asset=dict(h=1), models=[STATIC], description=root.name)
        (root / "declaration.json").write_text(json.dumps(document))
    known = dict(alpha={}, liquidity={}, volatility={}, matures={})
    shared, day = {}, 0
    for split, count in core.SESSIONS.items():
        rows = {name: [] for name in ("sample_id", "asset_id", "prediction_at", "target")}
        seen, hidden = [], []
        for session in range(count):
            instant = core.START + (day + session) * core.DAY
            z1, z2, z3, noise = rng.standard_normal((4, core.ASSETS))
            rows["sample_id"] += [f"{split[0]}{session:03d}-{a:02d}" for a in range(core.ASSETS)]
            rows["asset_id"] += [f"US/A{a:02d}" for a in range(core.ASSETS)]
            rows["prediction_at"] += [instant] * core.ASSETS
            rows["target"] += list(0.3 * z1 + 0.2 * z2 + 0.4 * z3 + noise)
            seen.append(z1)
            hidden.append(z3)
        day += count + 2
        shared[split] = (rows, np.concatenate(seen), np.concatenate(hidden))
        ids, size = rows["sample_id"], len(rows["sample_id"])
        known["alpha"][split] = dict(zip(ids, 0.002 * rng.standard_normal(size), strict=True))
        known["liquidity"][split] = dict(zip(ids, rng.normal(15, 2, size), strict=True))
        known["volatility"][split] = dict(zip(ids, rng.normal(3, 0.5, size), strict=True))
        known["matures"][split] = dict(
            zip(ids, np.asarray(rows["prediction_at"], dtype=np.int64) + core.DAY, strict=True)
        )
    for seed in SEEDS:
        for model in ("titans_mac_nogain", "titans_mac", STATIC):
            folder = state / "runs" / model / str(core.YEAR) / f"seed-{seed}"
            scale = static_noise if model == STATIC else core_noise
            for split, (rows, z1, _) in shared.items():
                core._write(folder, split, rows, z1 + scale * rng.standard_normal(len(z1)))
            record = dict(
                status="completed",
                final_test_opened=False,
                run_id=f"{model}-{seed}",
                request=dict(view_sha256="f" * 64),
            )
            (folder / "run.json").write_text(json.dumps(record))
        static = state / "runs" / STATIC / str(core.YEAR) / f"seed-{seed}"
        organ = roots[seed] / "runs" / f"{STATIC}__{arm}" / str(core.YEAR) / f"seed-{seed}"
        for split in og.AGGREGATED:
            rows, _, z3 = shared[split]
            own = pq.read_table(static / f"{split}-predictions.parquet").column("prediction")
            extra = organ_strength * z3 + 0.3 * rng.standard_normal(len(z3))
            core._write(organ, split, rows, own.to_numpy() + extra)
        record = dict(
            status="completed",
            final_test_opened=False,
            model=f"{STATIC}__{arm}",
            origin=core.YEAR,
            seed=seed,
            arm=arm,
            parent=dict(model=STATIC, run_id=f"{STATIC}-{seed}"),
            inputs=dict(
                emissions={
                    s: pilots._sha256(static / f"{s}-predictions.parquet") for s in og.SPLITS
                }
            ),
            outputs={s: pilots._sha256(organ / f"{s}-predictions.parquet") for s in og.AGGREGATED},
        )
        (organ / "run.json").write_text(json.dumps(record))
    manifest = state / "cohorts" / str(core.YEAR) / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    return dict(state=state, roots=roots, known=known)


def _arguments(paths, output, *extra, declaration=DECLARATION):
    arguments = ["medir", "--declaration", str(declaration), "--state", str(paths["state"])]
    arguments += ["--origin", str(core.YEAR), "--output", str(output)]
    for seed, root in sorted(paths["roots"].items()):
        arguments += ["--organs", f"{seed}={root}"]
    return [*arguments, *extra]


def _expected_c3(paths):
    tables = [
        pq.read_table(
            paths["state"]
            / "runs"
            / "titans_mac_nogain"
            / str(core.YEAR)
            / f"seed-{s}"
            / "evaluation-predictions.parquet"
        )
        for s in SEEDS
    ]
    instants = tables[0].column("prediction_at").cast("int64").to_numpy()
    target = tables[0].column("target").to_numpy()
    values = []
    for instant in np.unique(instants):
        rows = instants == instant
        ranks = np.mean(
            [rankdata(t.column("prediction").to_numpy()[rows]) / rows.sum() for t in tables], axis=0
        )
        values.append(spearmanr(ranks, target[rows])[0])
    return float(np.mean(values))


def test_f6_of_an_origin(tmp_path, monkeypatch, capsys):
    paths = campaign(tmp_path)
    monkeypatch.setattr(holo.ac, "known_rows", core._known(paths))
    output = tmp_path / "holografico"
    assert holo.main(_arguments(paths, output)) == 0
    folder = output / str(core.YEAR)
    summary = json.loads((folder / "resumen.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and len(summary["organ_roots"]) == 2
    assert summary["verdict"]["enters"] and summary["verdict"]["organs_matter"]
    level = summary["levels"]["label"]["evaluation"]["C3"]["mean"]
    assert level == pytest.approx(_expected_c3(paths), abs=1e-9)
    gain = summary["contrasts"]["label"]["c3h3_vs_c3"]
    assert gain["mean"] >= 0.002 and gain["t_nw"] >= 2
    assert set(summary["contrasts"]) == {"label", "operable"}
    assert -1 <= summary["diversity"]["label"]["within_c3_h3"] <= 1
    assert set(summary["operable_factors"]) == {str(s) for s in SEEDS}
    sessions = pq.read_table(folder / "sesiones.parquet")
    assert set(sessions.column("series").to_pylist()) >= {"C3", "H3", "E3", "C3H3", "C3E3"}
    assert "Lectura de la regla" in (folder / "resumen.md").read_text(encoding="utf-8")
    capsys.readouterr()
    assert holo.main(_arguments(paths, output)) == 0
    assert "ya calculado" in capsys.readouterr().out


def test_organs_that_see_nothing_new_do_not_count(tmp_path, monkeypatch):
    # Un estático mucho mejor que el núcleo mejora el conjunto por sí solo; los órganos no ven nada
    # nuevo, así que el control a igual número de redes se lleva la ganancia.
    paths = campaign(tmp_path, organ_strength=0.0, core_noise=2.0, static_noise=0.3)
    monkeypatch.setattr(holo.ac, "known_rows", core._known(paths))
    assert holo.main(_arguments(paths, tmp_path / "h")) == 0
    summary = json.loads((tmp_path / "h" / str(core.YEAR) / "resumen.json").read_text())
    assert not summary["verdict"]["organs_matter"]
    second = summary["contrasts"]["label"]["c3e3_vs_c3"]
    assert second["mean"] > 0.002 and second["t_nw"] >= 2


def test_mixed_organ_declarations_and_missing_inputs_fail_with_a_reason(
    tmp_path, monkeypatch, capsys
):
    paths = campaign(tmp_path)
    monkeypatch.setattr(holo.ac, "known_rows", core._known(paths))
    other = paths["roots"][43] / "declaration.json"
    other.write_text(json.dumps(dict(kind="organos", asset=dict(h=2), models=[STATIC])))
    assert holo.main(_arguments(paths, tmp_path / "h")) == 1
    assert "no comparten declaración" in capsys.readouterr().err
    document = dict(kind="organos", asset=dict(h=1), models=[STATIC], description="b")
    other.write_text(json.dumps(document))
    record = (
        paths["roots"][42]
        / "runs"
        / f"{STATIC}__combined"
        / str(core.YEAR)
        / "seed-42"
        / "run.json"
    )
    changed = json.loads(record.read_text())
    changed["parent"]["run_id"] = "otro-ajuste"
    record.write_text(json.dumps(changed))
    assert holo.main(_arguments(paths, tmp_path / "h")) == 1
    assert "no es el brazo combinado" in capsys.readouterr().err
    organ = paths["roots"][44] / "runs" / f"{STATIC}__combined" / str(core.YEAR) / "seed-44"
    (organ / "evaluation-predictions.parquet").unlink()
    assert holo.main(_arguments(paths, tmp_path / "h")) == og.MISSING_EXIT
    assert "seed-44" in capsys.readouterr().err


@pytest.mark.parametrize("arm", sorted(holo.WP5_SHA256))
def test_the_wp5_variants_differ_from_f6_only_in_the_organ_and_the_description(arm):
    variant = json.loads(WP5[arm].read_text(encoding="utf-8"))
    kept = ("organ", "description")
    assert {k: v for k, v in variant.items() if k not in kept} == {
        k: v for k, v in declaration().items() if k not in kept
    }
    assert variant["organ"] == f"{STATIC}__{arm}"
    _, digest = holo.checked_declaration(variant)
    assert digest == holo.WP5_SHA256[arm] and holo.REGISTERED[digest] == arm
    assert holo.organ_arm(variant) == arm and holo.organ_arm(declaration()) == "combined"


def test_a_variant_with_another_organ_or_rule_is_refused():
    variant = json.loads(WP5["dense_bank"].read_text(encoding="utf-8"))
    variant["organ"] = f"{STATIC}__asset_trace"
    with pytest.raises(ValueError, match="no es la registrada para F6"):
        holo.checked_declaration(variant)
    variant = json.loads(WP5["asset_trace"].read_text(encoding="utf-8"))
    variant["rule"]["min_gain"] = 0.001
    with pytest.raises(ValueError, match="no es la registrada para F6"):
        holo.checked_declaration(variant)


def test_a_wp5_variant_measures_its_own_organ_with_the_rule_of_f6(tmp_path, monkeypatch, capsys):
    paths = campaign(tmp_path, arm="dense_bank")
    monkeypatch.setattr(holo.ac, "known_rows", core._known(paths))
    output = tmp_path / "holografico-wp5"
    assert holo.main(_arguments(paths, output, declaration=WP5["dense_bank"])) == 0
    summary = json.loads((output / str(core.YEAR) / "resumen.json").read_text(encoding="utf-8"))
    assert summary["declaration_sha256"] == holo.WP5_SHA256["dense_bank"]
    assert summary["verdict"]["enters"] and summary["verdict"]["organs_matter"]
    assert summary["levels"]["label"]["evaluation"]["C3"]["mean"] == pytest.approx(
        _expected_c3(paths), abs=1e-9
    )
    # F6 busca su brazo combinado, que esta raíz no tiene.
    assert holo.main(_arguments(paths, tmp_path / "f6")) == og.MISSING_EXIT
    capsys.readouterr()
    run = paths["roots"][42] / "runs" / f"{STATIC}__dense_bank" / str(core.YEAR) / "seed-42"
    record = json.loads((run / "run.json").read_text())
    record["arm"] = "combined"
    (run / "run.json").write_text(json.dumps(record))
    arguments = _arguments(paths, tmp_path / "otra", declaration=WP5["dense_bank"])
    assert holo.main(arguments) == 1
    assert "no es el brazo dense_bank" in capsys.readouterr().err
