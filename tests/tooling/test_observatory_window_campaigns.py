"""Las campañas por ventanas de A y A v2 llegan al observatorio con sus cuatro etapas.

Los resúmenes salen de las propias etapas: la campaña base y la etapa de políticas de
`rl_stage_fixture`, que no ajustan ninguna red, y la etapa de adaptadores con su plan real
y su escritor de resúmenes, sin ejecutar ajustes. Las declaraciones de
`configs/observatory/campaigns.json` se comprueban con los planes reales de cada etapa.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_titan.observatory import collector as collector_module
from mars_titan.observatory import window_campaigns
from mars_titan.observatory.collector import KINDS, Collector, digest, write_pages
from mars_titan.observatory.live_server import LABEL, Limits
from mars_titan.observatory.publication import GitPublisher
from mars_titan.observatory.window_campaigns import arm_model, campaign_state
from mars_titan.posttraining import campaign_stage as adapter_stage
from mars_titan.posttraining import staged_chain
from mars_titan.simulation import campaign_stage as policy_stage
from mars_titan.simulation import policy_plan
from mars_titan.training import campaign_plan, masked_campaign, modality_ablation_stage
from scripts.serve_observatory import declared_campaigns
from tests.simulation import rl_stage_fixture

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/observatory/campaigns.json"
ADAPTERS_A_V2 = ROOT / "configs/posttraining/historical-masked-adapter-stage-a-v2.json"
SOURCES = [
    source
    for source in json.loads(CONFIG.read_text())["sources"]
    if source["kind"] == collector_module.WINDOW_SOURCE
]
STATE_KEYS = {
    "id",
    "kind",
    "stage",
    "status",
    "updated_at",
    "summary_modified_at",
    "planned",
    "completed",
    "final_test_opened",
    "vocabulary",
    "models",
    "cells",
    "active",
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def planned_jobs(source):
    """Identificadores de trabajo del plan real de la etapa, sin leer datos."""
    path = ROOT / source["configuration"]
    if source["stage"] == "base":
        return [job["id"] for job in campaign_plan.plan_campaign(campaign_plan.load_campaign(path))]
    if source["stage"] == "adapters":
        stage = adapter_stage.load_stage(path)
        jobs = adapter_stage.plan_stage(stage)
        return [job["id"] for job in [*jobs, *adapter_stage.plan_chain(stage, jobs)]]
    if source["stage"] == "ablation":
        stage = modality_ablation_stage.load_stage(path)
        return [job["id"] for job in modality_ablation_stage.plan_stage(stage)]
    return [job["id"] for job in policy_plan.plan_stage(policy_plan.load_stage(path))]


def test_stage_kinds_are_the_ones_the_stages_write():
    assert window_campaigns.STAGES == {
        masked_campaign.RUN_KIND: ("base", 4),
        adapter_stage.RUN_KIND: ("adapters", 4),
        modality_ablation_stage.RUN_KIND: ("ablation", 4),
        policy_stage.RUN_KIND: ("policies", 6),
    }


def test_campaigns_a_and_a_v2_declare_their_four_stages_for_both_modes():
    stages = ("base", "adapters", "ablation", "policies")
    assert [(s["id"], s["stage"]) for s in SOURCES] == [
        (f"historical-masked-{variant}-{stage}", stage)
        for variant in ("a", "a-v2")
        for stage in stages
    ]
    # El servidor en directo sigue como mucho ocho campañas y exige etiquetas cortas.
    assert len(SOURCES) <= 8 and all(LABEL.fullmatch(s["id"]) for s in SOURCES)
    assert len({s["path"] for s in SOURCES}) == len(SOURCES)
    assert all(s["path"].startswith("data/interim/") and s["domain"] == "real" for s in SOURCES)


@pytest.mark.parametrize("source", SOURCES, ids=lambda source: source["id"])
def test_every_planned_job_of_the_declared_stage_is_read_and_has_a_catalog_model(source, tmp_path):
    jobs = planned_jobs(source)
    assert len(jobs) == len(set(jobs)) and len(jobs) <= collector_module.WINDOW_JOBS
    assert len(jobs) <= Limits().max_campaign_jobs
    kind = next(k for k, (stage, _) in window_campaigns.STAGES.items() if stage == source["stage"])
    dump(
        tmp_path / "summary.json",
        dict(kind=kind, status="running", jobs=dict.fromkeys(jobs, False)),
    )
    state = campaign_state(source["id"], tmp_path)
    assert state["stage"] == source["stage"] and len(state["cells"]) == len(jobs)
    unknown = sorted(arm for arm, model in state["models"].items() if model not in KINDS)
    assert unknown == []
    assert "unknown" not in state["models"].values()
    # Un evento SSE admite 4 MiB y el sitio lee documentos de hasta 8 MiB.
    assert len(json.dumps(dict(available=True, **state), separators=(",", ":"))) < 4 * 1024**2


def test_arms_of_every_family_map_to_their_model():
    expected = {
        "gru": "gru",
        "gru_episodic": "gru_episodic",
        "transformer_compact_online": "transformer_compact",
        "titans_transformer_direct": "titans_mac",
        "titans_mac_online__head_readout": "titans_mac",
        "mars_titan_m1_k4__chain": "mars_titan",
        "cm_v1_core_b": "cm_v1",
        "cm_v1_bcm__frozen_parent": "cm_v1",
        "xgboost__chain": "xgboost",
        "gru/klpo_terminal": "klpo",
        "mars_titan_m2/ppo_clip_full_kl": "ppo",
        "cm_v1_b/market_index": "market_index",
        "rnn/equal_weight_monthly": "equal_weight_monthly",
        "grux": "unknown",
        "ppox": "unknown",
    }
    assert {arm: arm_model(arm) for arm in expected} == expected


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return rl_stage_fixture.base_campaign(tmp_path_factory.mktemp("window-base"), "A")


def test_the_base_campaign_summary_is_read_with_every_receipt(base):
    state = campaign_state("base", base.output)
    assert set(state) == STATE_KEYS
    assert state["stage"] == "base" and state["status"] == "completed"
    assert {cell[4] for cell in state["cells"]} == {"done"}
    assert all(cell[5].endswith("Z") for cell in state["cells"])
    assert state["models"] == {"gru": "gru", "lstm": "lstm"}
    assert state["updated_at"].endswith("Z") and state["final_test_opened"] is False


def test_the_policy_stage_summary_is_read_with_markets_predictors_and_open_runs(
    base, tmp_path, learning_doubles
):
    output = tmp_path / "policies"
    # El primer ajuste del plan, el mismo que pausa `tests/simulation/test_campaign_stage.py`.
    first = "US/US/fold-002/gru/klpo_terminal/fit-s42"
    paused = rl_stage_fixture.run(base, output, rl_stage_fixture.ScriptedLearner(pause={first}))
    assert paused["status"] == "paused"
    state = campaign_state("policies", output)
    assert state["stage"] == "policies" and state["status"] == "paused"
    states = {cell[4] for cell in state["cells"]}
    assert states == {"attempt", "pending"}
    # La etapa de políticas guarda el intento en `run/`, sin informe por épocas.
    assert state["active"] == [
        dict(job=first, attempt="run", updated_at=None, global_step=None, epochs=[])
    ]
    summary = rl_stage_fixture.run(base, output, rl_stage_fixture.ScriptedLearner())
    assert summary["status"] == "completed" and summary["metrics"]
    state = campaign_state("policies", output)
    assert set(state) == STATE_KEYS and state["active"] == []
    assert {cell[4] for cell in state["cells"]} == {"done"}
    assert all(cell[5].endswith("Z") for cell in state["cells"])
    # Ámbito con mercado y brazo con predictor, como `ámbito/mercado/ventana/predictor/brazo`.
    assert state["vocabulary"]["scopes"] == ["US/US"]
    arms = state["vocabulary"]["arms"]
    assert {"gru/klpo_terminal", "gru/double_dqn", "lstm/market_index"} <= set(arms)
    assert state["models"]["gru/klpo_terminal"] == "klpo"
    assert state["models"]["lstm/equal_weight_monthly"] == "equal_weight_monthly"
    assert state["planned"] == summary["planned"] and state["completed"] == summary["completed"]
    # Las métricas financieras del resumen no salen del observatorio.
    assert "metrics" not in json.dumps(state)


def adapter_output(root):
    """Salida de la etapa de adaptadores de A v2 escrita con su plan y su resumen reales.

    No se ejecuta ningún ajuste: los recibos, la selección de la cadena y los intentos
    abiertos son archivos en las rutas que usa la etapa, y el resumen lo escribe su propio
    `_summary`.
    """
    stage = adapter_stage.load_stage(ADAPTERS_A_V2)
    jobs = adapter_stage.plan_stage(stage)
    chains = adapter_stage.plan_chain(stage, jobs)
    fit = next(job for job in jobs if job["kind"] == "fit" and job["window"] == "fold-001")
    running = next(job for job in jobs if job["kind"] == "fit" and job["id"] != fit["id"])
    frozen = next(job for job in jobs if job["kind"] != "fit" and job["family"] == "titans_mac")
    chain = chains[0]
    dump(root / "jobs" / fit["id"] / "receipt.json", dict(status="completed"))
    selection = staged_chain.chain_folder(
        root, chain["scope"], chain["window"], chain["base_arm"], chain["seed"]
    )
    dump(selection / staged_chain.SELECTION, dict(kind=staged_chain.SELECTION_KIND))
    report = dict(
        global_step=12,
        epochs=[
            dict(
                epoch=1,
                train=dict(mae=0.021, samples_per_second=33300.0, elapsed_seconds=40.5),
                validation=dict(mae=0.02, session_mae=0.019),
            )
        ],
    )
    dump(root / "jobs" / running["id"] / "run" / "run.json", report)
    (root / "jobs" / frozen["id"] / "attempt-0001").mkdir(parents=True)
    receipts = {fit["id"]: dict(updates=7)}
    selections = {chain["id"]: dict(selected=dict(kind="base", job=fit["id"]))}
    state = SimpleNamespace(
        receipts=receipts, selections=selections, budgets={}, released_indices={}
    )
    adapter_stage._summary(root, dict(fixture=True), jobs, chains, state, "running")
    return SimpleNamespace(
        jobs=len(jobs) + len(chains), fit=fit, running=running, frozen=frozen, chain=chain
    )


def test_the_adapter_stage_summary_confirms_chain_selections_by_their_selection_file(tmp_path):
    written = adapter_output(tmp_path)
    state = campaign_state("adapters", tmp_path)
    assert state["stage"] == "adapters" and len(state["cells"]) == written.jobs
    vocabulary = state["vocabulary"]
    by_id = {
        "/".join(
            (
                vocabulary["scopes"][cell[0]],
                vocabulary["windows"][cell[1]],
                vocabulary["arms"][cell[2]],
                vocabulary["names"][cell[3]],
            )
        ): cell
        for cell in state["cells"]
    }
    for job in (written.fit, written.chain):
        assert by_id[job["id"]][4] == "done" and by_id[job["id"]][5].endswith("Z"), job["id"]
    assert by_id[written.running["id"]][4] == "attempt"
    assert by_id[written.frozen["id"]][4] == "attempt"
    assert sum(cell[4] != "pending" for cell in state["cells"]) == 4
    active = {entry["job"]: entry for entry in state["active"]}
    assert active[written.frozen["id"]]["attempt"] == "attempt-0001"
    assert active[written.running["id"]]["attempt"] == "run"
    assert active[written.running["id"]]["global_step"] == 12
    assert active[written.running["id"]]["epochs"] == [
        dict(
            epoch=1,
            train_mae=0.021,
            mae=0.02,
            session_mae=0.019,
            train_samples_per_second=33300.0,
            train_seconds=40.5,
        )
    ]
    assert state["models"][f"{written.chain['base_arm']}__chain"] != "unknown"
    assert state["completed"] == dict(training_jobs=1, prediction_jobs=0, selection_jobs=1)


@pytest.mark.parametrize(
    "summary",
    [
        dict(kind=policy_stage.RUN_KIND, jobs={"US/fold-000/gru/fit-s42": False}),
        dict(kind=masked_campaign.RUN_KIND, jobs={"US/US/fold-000/gru/klpo/fit-s42": False}),
        dict(kind="other", jobs={"US/fold-000/gru/a": False, "US/US/fold-000/gru/b/c": False}),
        dict(kind=masked_campaign.RUN_KIND, jobs={"US/fold-000/gru/fit-s42": 1}),
        dict(kind=masked_campaign.RUN_KIND, jobs={"US/../gru/fit-s42": False}),
        dict(kind=masked_campaign.RUN_KIND, jobs={"US/fold-000/gru": False}),
        dict(kind=masked_campaign.RUN_KIND, jobs=["US/fold-000/gru/fit-s42"]),
    ],
)
def test_a_summary_outside_the_job_contract_is_rejected(summary, tmp_path):
    dump(tmp_path / "summary.json", summary)
    with pytest.raises(ValueError):
        campaign_state("x", tmp_path)


def test_more_jobs_than_the_limit_are_rejected(tmp_path):
    jobs = {f"US/fold-000/gru/fit-s{seed}": False for seed in range(5)}
    dump(tmp_path / "summary.json", dict(kind=masked_campaign.RUN_KIND, jobs=jobs))
    assert len(campaign_state("x", tmp_path, max_jobs=5)["cells"]) == 5
    with pytest.raises(ValueError):
        campaign_state("x", tmp_path, max_jobs=4)


def write_root(root):
    """Raíz con dos campañas por ventanas declaradas: una con resumen y otra sin empezar."""
    for name in ("policies.json", "adapters.json"):
        dump(root / "configs" / name, dict(fixture=True))
    jobs = {
        "US+CN/US/fold-001/gru/klpo_terminal/fit-s42": True,
        "US+CN/US/fold-001/gru/cash/reference": False,
        "US+CN/CN/fold-001/mars_titan_m1/ppo_clip_full_kl/fit-s42": False,
    }
    folder = root / "data/policies"
    dump(folder / "jobs" / next(iter(jobs)) / "receipt.json", dict(status="completed"))
    (folder / "jobs/US+CN/CN/fold-001/mars_titan_m1/ppo_clip_full_kl/fit-s42/run").mkdir(
        parents=True
    )
    dump(
        folder / "summary.json",
        dict(
            kind=policy_stage.RUN_KIND,
            status="running",
            planned=dict(fit=2, reference=1),
            completed=dict(fit=1),
            metrics=dict(secret_sharpe=3.14159),
            jobs=jobs,
            final_test_opened=False,
            updated_at_utc="2026-10-10T12:00:00+00:00",
        ),
    )
    return [
        dict(
            id="policies",
            path="data/policies",
            kind="window_campaign",
            domain="real",
            stage="policies",
            configuration="configs/policies.json",
        ),
        dict(
            id="adapters",
            path="data/adapters",
            kind="window_campaign",
            domain="real",
            stage="adapters",
            configuration="configs/adapters.json",
        ),
    ]


def test_the_collector_publishes_window_campaigns_as_documents_and_not_as_records(tmp_path):
    root, output = tmp_path / "root", tmp_path / "public"
    sources = write_root(root)
    with Collector(root, tmp_path / "cache.sqlite") as collector:
        snapshot = collector.collect(sources, now="2026-10-10T12:00:05+00:00")
    assert snapshot["runs"] == [] and snapshot["campaigns"] == []
    index = write_pages(snapshot, output)
    entries = {entry["id"]: entry for entry in index["window_campaigns"]}
    assert entries["adapters"] == dict(
        id="adapters",
        domain="real",
        stage="adapters",
        configuration="configs/adapters.json",
        path=None,
        status=None,
        updated_at=None,
        jobs=0,
        done=0,
        attempts=0,
    )
    policies = entries["policies"]
    assert (policies["jobs"], policies["done"], policies["attempts"]) == (3, 1, 1)
    assert policies["status"] == "running" and policies["updated_at"] == "2026-10-10T12:00:00Z"
    document = json.loads((output / policies["path"]).read_text())
    assert policies["path"] == f"windows/{digest(document)}.json"
    assert document["stage"] == "policies" and document["active"][0]["attempt"] == "run"
    assert document["models"]["mars_titan_m1/ppo_clip_full_kl"] == "ppo"
    published = (output / policies["path"]).read_text() + (output / "observatory.json").read_text()
    assert "secret_sharpe" not in published and "3.14159" not in published
    assert {model["id"] for model in index["models"]} >= set(document["models"].values())
    # Ni el índice ni las páginas llevan la matriz completa.
    assert "cells" not in json.dumps(index)


@pytest.mark.parametrize(
    "change",
    [
        lambda sources: sources[0].update(stage="adapters"),
        lambda sources: sources[0].update(configuration="configs/missing.json"),
        lambda sources: sources[0].pop("stage"),
        lambda sources: sources[0].update(stage="training"),
    ],
)
def test_a_window_campaign_must_match_its_declared_stage_and_configuration(change, tmp_path):
    sources = write_root(tmp_path / "root")
    change(sources)
    with Collector(tmp_path / "root", tmp_path / "cache.sqlite") as collector:
        with pytest.raises(ValueError):
            collector.collect(sources)


def test_an_unchanged_summary_is_reused_until_the_refresh_interval(tmp_path, monkeypatch):
    root = tmp_path / "root"
    sources = write_root(root)
    calls = []
    real = collector_module.campaign_state

    def counted(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    clock = [1000.0]
    monkeypatch.setattr(collector_module, "campaign_state", counted)
    monkeypatch.setattr(
        collector_module, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
    )
    with Collector(root, tmp_path / "cache.sqlite") as collector:
        collector.collect(sources)
        clock[0] += collector_module.WINDOW_REFRESH_SECONDS - 1
        collector.collect(sources)
        assert calls == ["policies"]
        # Un intento nuevo aparece sin reescribir el resumen: se ve al vencer el intervalo.
        clock[0] += 1
        snapshot = collector.collect(sources)
        assert calls == ["policies", "policies"]
        summary = root / "data/policies/summary.json"
        value = json.loads(summary.read_text())
        value["jobs"]["US+CN/US/fold-001/gru/cash/reference"] = True
        summary.write_text(json.dumps(value))
        snapshot = collector.collect(sources)
    assert calls == ["policies"] * 3
    states = [cell[4] for cell in snapshot["window_campaigns"][0]["state"]["cells"]]
    assert states.count("done") == 2


def test_documents_that_the_index_no_longer_lists_are_removed_after_the_grace(tmp_path):
    output = tmp_path / "public"
    snapshot = {"schema_version": 2, "project": "MARS-TITAN", "runs": []}
    old = output / "windows" / f"{'a' * 64}.json"
    recent = output / "windows" / f"{'b' * 64}.json"
    for path in (old, recent):
        dump(path, {})
    past = time.time() - collector_module.WINDOW_GRACE_SECONDS - 5
    os.utime(old, (past, past))
    write_pages(snapshot, output)
    assert not old.exists() and recent.exists()


def test_the_publisher_sends_window_documents_and_prunes_what_the_index_dropped(tmp_path):
    def git(cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)

    remote, checkout, output = (tmp_path / name for name in ("remote.git", "checkout", "public"))
    checkout.mkdir()
    git(tmp_path, "init", "--bare", str(remote))
    git(checkout, "init", "-b", "observatory-data")
    git(checkout, "config", "user.name", "Prueba local")
    git(checkout, "config", "user.email", "test@example.invalid")
    git(checkout, "remote", "add", "origin", str(remote))
    stale = f"pages/{'c' * 64}.json"
    dump(checkout / stale, {})
    git(checkout, "add", stale)
    git(checkout, "commit", "-m", "chore: older page")
    root = tmp_path / "root"
    sources = write_root(root)
    with Collector(root, tmp_path / "cache.sqlite") as collector:
        index = write_pages(collector.collect(sources), output)
    window = next(entry["path"] for entry in index["window_campaigns"] if entry["path"])
    GitPublisher(checkout, dispatch=False).publish(output)
    tracked = set(git(checkout, "ls-files").stdout.split())
    assert tracked == {"observatory.json", window}
    # Un documento alterado no coincide con su huella y no se publica.
    (output / window).write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError):
        GitPublisher(checkout, dispatch=False).publish(output)


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Falta Node.js para el validador de la web"
)
def test_collector_output_passes_the_site_validators(tmp_path):
    root, output = tmp_path / "root", tmp_path / "public"
    with Collector(root, tmp_path / "cache.sqlite") as collector:
        index = write_pages(collector.collect(write_root(root)), output)
    window = next(entry["path"] for entry in index["window_campaigns"] if entry["path"])
    state = ROOT / "site" / "state.mjs"
    javascript = """
        import { readFileSync } from 'node:fs';
        const { validateSnapshot, validateWindowCampaign } = await import(process.argv[1]);
        const index = validateSnapshot(JSON.parse(readFileSync(process.argv[2], 'utf8')));
        const raw = JSON.parse(readFileSync(process.argv[3], 'utf8'));
        const document = validateWindowCampaign(raw);
        process.stdout.write(JSON.stringify({ entries: index.window_campaigns, document }));
    """
    validated = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            javascript,
            state.as_uri(),
            str(output / "observatory.json"),
            str(output / window),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert validated.returncode == 0, validated.stderr
    result = json.loads(validated.stdout)
    assert [entry["id"] for entry in result["entries"]] == ["policies", "adapters"]
    assert result["document"]["stage"] == "policies"
    assert len(result["document"]["cells"]) == 3


def test_the_live_server_follows_the_declared_window_campaigns():
    declared = declared_campaigns(CONFIG, ROOT)
    assert list(declared) == [source["id"] for source in SOURCES]
    assert (
        declared["historical-masked-a-v2-policies"]
        == ROOT / "data/interim/historical-masked-a-v2/policies"
    )
