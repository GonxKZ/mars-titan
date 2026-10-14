"""Retención v3 sobre la campaña A reducida, con la etapa de adaptadores dentro de la ventana.

La campaña de `campaign_fixture` (dos ventanas US, GRU con la semilla 42 y pesos iniciales)
se completa con su etapa de adaptadores, cuyo optimizador solo registra gradientes, y se
calculan desde las filas los informes que leerá la publicación. Después la retención v3
recorre una copia: la etapa se vuelve a recorrer en su ventana con la compactación al
confirmar y el cierre de bloques, que guarda los agregados de la GRU y regenera y libera sus
tablas. Se comprueba que todo se libera solo con regeneraciones idénticas, que queda un estado
por ajuste y ninguna caché, y que la publicación desde los agregados sale igual que la que
leía las filas. Ningún paso cambia pesos y 2024 no se abre. Los puntos de entrada se admiten
con los dobles de `tests/posttraining/conftest.py`, sin rebajar la protección real.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from mars_titan.data import prediction_files
from mars_titan.evaluation import comparison_matrix
from mars_titan.evaluation import session_table_contrasts as tables
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.posttraining import campaign_stage
from mars_titan.posttraining import stage_comparison as compare
from mars_titan.training import masked_campaign as engine
from mars_titan.training import rolling_retention as rolling
from tests.posttraining.campaign_fixture import CpuLease, base_campaign
from tests.posttraining.real_only import real_data_only
from tests.posttraining.test_publication_from_aggregates import (
    PLAIN_METRICS,
    REGENERATORS,
    RUNNING,
    assert_published_from_aggregates,
    base_tables,
    plain,
    publish,
)
from tests.training import publication_fixture
from tests.training.test_rolling_retention import schedule

ROOT = Path(__file__).resolve().parents[2]
WINDOW = "fold-001"


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return base_campaign(tmp_path_factory.mktemp("retention-v3"), "A", comparison=plain)


@pytest.fixture
def retention(tmp_path):
    """Retención v3 sin las políticas como consumidoras."""
    document = json.loads(
        (ROOT / "configs/baselines/historical-masked-retention-v3.json").read_text()
    )
    document["consumers"] = {}
    path = tmp_path / "retention.json"
    path.write_text(json.dumps(document))
    return path


def stage(base, campaign, adapters, **options):
    with real_data_only():
        return campaign_stage.run_stage(
            base.stage,
            base.views,
            campaign,
            adapters,
            lease=CpuLease,
            stop=RUNNING,
            device="cpu",
            **options,
        )


def expected(base, adapters, declared, folder):
    """Informes de la campaña y de la etapa calculados desde las filas, antes de liberar."""
    primary = engine.write_sources(base.campaign, base.views, base.output, "US")
    comparison = Path(base.campaign).parent / "comparison.json"
    walked = walk.write_walk_forward(comparison, primary, "US", folder / "walk-forward")
    loaded = comparison_matrix.load_matrix(declared.matrix)
    manifest = tables.write_sources(
        folder / "matrix-sources.json",
        "US",
        [dict(kind=tables.WALK_SOURCE, path=folder / "walk-forward" / "comparison.json")],
        views=loaded["views"],
        config=loaded["comparison_config"],
    )
    sources = compare.write_sources(
        declared.stage, "US", "gru", base_sources=primary, stage_output=adapters
    )
    _, report, sessions, _ = compare.evaluate(declared.stage, sources, "US", "gru")
    return SimpleNamespace(
        base=base,
        declared=declared,
        walk=(walked, pq.read_table(folder / "walk-forward" / "sessions.parquet")),
        matrix=comparison_matrix.evaluate(declared.matrix, manifest, "US"),
        stage=(report, sessions, sources),
    )


def test_v3_releases_the_adapter_rows_by_blocks_and_publishes_the_same_comparison(
    base, retention, tmp_path, recorder
):
    adapters = tmp_path / "confirmed"
    assert stage(base, base.output, adapters)["status"] == "completed"
    created = len(recorder.optimizers)
    declared = publication_fixture.declare(
        tmp_path / "declared", base.campaign, stage=base.stage, metrics=PLAIN_METRICS
    )
    reports = expected(base, adapters, declared, tmp_path / "expected")
    # Copias propias: la liberación no toca la campaña compartida del módulo.
    output, walked_adapters = tmp_path / "campaign", tmp_path / "adapters"
    shutil.copytree(base.output, output, symlinks=True)
    shutil.copytree(adapters, walked_adapters, symlinks=True)
    path, campaign = schedule(base, tmp_path)
    state = rolling.Rolling(
        rolling.load_retention(retention),
        base.campaign,
        base.views,
        output,
        rolling.load_schedule(path, campaign),
        adapters=dict(stage=base.stage, output=walked_adapters),
        regenerators=REGENERATORS,
        publication=declared.publication,
        adapter_device="cpu",
        adapter_lease=CpuLease,
    )
    runners = dict(
        base=lambda window: dict(status="completed"),
        adapters=lambda window: stage(
            base,
            output,
            walked_adapters,
            window=window,
            compact_rows=state.folder / "rows",
            on_block=state.block,
        ),
    )
    assert rolling.run_rolling(state, runners)["status"] == "completed"
    # La etapa confirmada no vuelve a ajustar y la regeneración no crea optimizadores.
    assert len(recorder.optimizers) == created
    receipts = [json.loads(p.read_text()) for p in (walked_adapters / "jobs").rglob("receipt.json")]
    assert len(receipts) == 7
    for receipt in receipts:
        for item in receipt["predictions"].values():
            found = prediction_files.verify(walked_adapters / item["path"], item["sha256"])
            assert found == prediction_files.RELEASED
    decisions = sorted((state.folder / "regeneration-reports").glob("adapters_*.json"))
    assert len(decisions) == 7
    assert all(json.loads(p.read_text())["identical"] is True for p in decisions)
    for checkpoints in (walked_adapters / "jobs").rglob("checkpoints"):
        assert len(list(checkpoints.glob("state-*.pt"))) == 1, checkpoints
    assert not list((walked_adapters / "windows-data").rglob("predictions.sqlite*"))
    block = json.loads((state.folder / "blocks" / "US" / WINDOW / "gru.json").read_text())
    assert (block["released"], block["kept_for_policies"], block["not_regenerable"]) == (7, 0, 0)
    ledger = state.ledger()["windows"]
    assert ledger[WINDOW]["phases"]["adapters"]["blocks"]["count"] == 1
    written = ledger[WINDOW]["phases"]["aggregates"]["written"]["US"]["posttraining"]["gru"]
    assert written == block["aggregates"]
    assert set(base_tables(state, WINDOW).values()) == {prediction_files.RELEASED}
    destination = tmp_path / "published"
    receipt = publish(reports, state, destination)
    assert_published_from_aggregates(reports, state, destination, receipt)


def test_v3_stops_when_an_adapter_regeneration_differs(base, retention, tmp_path, recorder):
    """Una regeneración de adaptadores que no sale idéntica deja sus tablas compactadas y
    detiene el recorrido, también al reanudarlo, hasta que se revise."""
    adapters = tmp_path / "adapters"
    assert stage(base, base.output, adapters)["status"] == "completed"
    output = tmp_path / "campaign"
    shutil.copytree(base.output, output, symlinks=True)
    declared = publication_fixture.declare(
        tmp_path / "declared", base.campaign, stage=base.stage, metrics=PLAIN_METRICS
    )
    path, campaign = schedule(base, tmp_path)
    state = rolling.Rolling(
        rolling.load_retention(retention),
        base.campaign,
        base.views,
        output,
        rolling.load_schedule(path, campaign),
        adapters=dict(stage=base.stage, output=adapters),
        regenerators=REGENERATORS,
        publication=declared.publication,
        adapter_device="cpu",
        adapter_lease=CpuLease,
    )
    job = "US/fold-001/gru__head/fit-s42"
    regenerate = rolling.regeneration._regenerate_adapter

    def differs(state_, job_, destination):
        report = regenerate(state_, job_, destination)
        return dict(report, identical=report["identical"] and job_["id"] != job)

    runners = dict(
        base=lambda window: dict(status="completed"),
        adapters=lambda window: stage(
            base,
            output,
            adapters,
            window=window,
            compact_rows=state.folder / "rows",
            on_block=state.block,
        ),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rolling.regeneration, "_regenerate_adapter", differs)
        with pytest.raises(ValueError, match="no salen idénticas"):
            rolling.run_rolling(state, runners)
    receipt = json.loads((adapters / "jobs" / job / "receipt.json").read_text())
    for item in receipt["predictions"].values():
        found = prediction_files.verify(adapters / item["path"], item["sha256"])
        assert found == prediction_files.COMPACTED
    assert not state.done(WINDOW, "adapters")
    # La diferencia comparada es definitiva: reanudar vuelve a detenerse sin regenerar.
    with pytest.raises(ValueError, match="no salen idénticas"):
        rolling.run_rolling(state, runners)


@pytest.mark.parametrize("last", [1, 2])
def test_v3_keeps_a_chain_evaluation_compacted_until_its_last_reader(
    base, retention, tmp_path, recorder, last
):
    """Una evaluación de la cadena que leerá una política sale del bloque compactada. La
    libera, tras regenerarla, la liberación de la ventana de su último lector: la propia
    (`last` 1) o una posterior a la campaña reducida (`last` 2), en cuyo caso sigue al final."""
    adapters = tmp_path / "adapters"
    assert stage(base, base.output, adapters)["status"] == "completed"
    output = tmp_path / "campaign"
    shutil.copytree(base.output, output, symlinks=True)
    declared = publication_fixture.declare(
        tmp_path / "declared", base.campaign, stage=base.stage, metrics=PLAIN_METRICS
    )
    path, campaign = schedule(base, tmp_path)
    state = rolling.Rolling(
        rolling.load_retention(retention),
        base.campaign,
        base.views,
        output,
        rolling.load_schedule(path, campaign),
        adapters=dict(stage=base.stage, output=adapters),
        regenerators=REGENERATORS,
        publication=declared.publication,
        adapter_device="cpu",
        adapter_lease=CpuLease,
    )
    job = "US/fold-001/gru__head/fit-s42"
    assert state.positions["US", WINDOW] == 1

    def chain_keep(self, index, only=None):
        return {job: (last, [])} if only in (None, ("US", WINDOW, "gru")) else {}

    runners = dict(
        base=lambda window: dict(status="completed"),
        adapters=lambda window: stage(
            base,
            output,
            adapters,
            window=window,
            compact_rows=state.folder / "rows",
            on_block=state.block,
        ),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rolling.Rolling, "chain_keep", chain_keep)
        assert rolling.run_rolling(state, runners)["status"] == "completed"
    block = json.loads((state.folder / "blocks" / "US" / WINDOW / "gru.json").read_text())
    assert (block["released"], block["kept_for_policies"], block["not_regenerable"]) == (6, 1, 0)
    receipt = json.loads((adapters / "jobs" / job / "receipt.json").read_text())
    found = {
        prediction_files.verify(adapters / item["path"], item["sha256"])
        for item in receipt["predictions"].values()
    }
    released = state.ledger()["windows"][WINDOW]["phases"]["release"]["adapters"]
    decisions = sorted((state.folder / "regeneration-reports").glob("adapters_*.json"))
    if last == 1:
        assert found == {prediction_files.RELEASED}
        assert released == dict(released=1, kept_for_policies=0, not_regenerable=0)
        assert len(decisions) == 7
    else:
        assert found == {prediction_files.COMPACTED}
        assert released == dict(released=0, kept_for_policies=1, not_regenerable=0)
        assert len(decisions) == 6
    assert all(json.loads(p.read_text())["identical"] is True for p in decisions)
