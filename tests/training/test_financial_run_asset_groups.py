"""Recorrido cronológico con los grupos de activos de H4 (#783), sin pasos.

El corpus técnico declara una sesión ausente en todo el mercado, así que el lector monta las
ventanas de la v3.1 con el bit de presencia. El optimizador registrador guarda los gradientes y
no cambia pesos.
"""

import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory import financial_observations as observations
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.models.titans.asset_groups import AssetGroupsConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_financial_run import SELECTION, RecordingOptimizer
from tests.training.test_financial_run_session_objective import (
    explicit_fastpath,  # noqa: F401
    records,
)
from tests.training.test_historical_temporal import prepare

ASSETS = 4
GROUPS = AssetGroupsConfig(buckets=3, clusters=2, iterations=8)


@pytest.fixture(scope="module")
def streams(tmp_path_factory):
    root = tmp_path_factory.mktemp("asset-groups")
    fixture = historical_temporal_fixture(
        root / "source", assets=ASSETS, market_absent={"US": ["2022-12-01"]}
    )
    prepare(fixture, root / "views")
    dataset = CorpusDataset(root / "views/fold-000/manifest.json", input_policy=HISTORICAL_MASKED)
    result = {}
    for name, start, end, _ in dataset.temporals["US"].partitioner.bounds:
        if name in ("train", "validation"):
            phase = FinancialPhase(name, int(start), int(start), int(end), int(end))
            index = observations.prepare_observation_index(
                dataset, root / f"index-{name}", phase=phase
            )
            result[name] = observations.FinancialObservationSource(dataset, index)
    return result


def trainer(sources, output, **options):
    config = FinancialConfig(
        sources["train"].specification(), hidden_size=32, variant="mac_online", asset_groups=GROUPS
    )
    recipe = dict(truncation=2, epochs=1, block_rows=8, selection=SELECTION) | options
    return ChronologicalTrainer(
        FinancialPredictor(config, dtype=torch.float64),
        ChronologicalRecipe(**recipe),
        train=sources["train"],
        validation=sources["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


def test_each_update_reads_whole_sessions_with_their_presence_windows(streams, tmp_path):
    assert streams["train"].specification().dimensions["prices"] == 6
    engine = trainer(streams, tmp_path / "run")
    seen = []
    original = engine.predictor.prepare

    def spy(batch, state, **options):
        seen.append(
            (
                len(set(batch.prediction_at)),
                len(batch.flow_ids),
                options.get("session_complete"),
                tuple(batch.inputs["prices"].shape[1:]),
            )
        )
        return original(batch, state, **options)

    engine.predictor.prepare = spy
    before = engine.predictor._parameter_id
    report = engine.run()
    assert report["status"] == "completed"
    assert seen and all(
        instants == 1 and rows == ASSETS and whole is True and shape == (64, 6)
        for instants, rows, whole, shape in seen
    )
    gradients = records(engine)
    assert gradients
    # W_o empieza en cero: en la primera actualización solo ella recibe gradiente de H4.
    first = gradients[0]
    assert first["asset_groups.out.weight"].abs().sum() > 0
    assert first["asset_groups.inner.weight"].abs().sum() == 0
    assert first["asset_groups.outer.weight"].abs().sum() == 0
    assert engine.predictor._parameter_id == before
    code = engine.identity["implementation"]
    assert {"mars_titan.models.titans.asset_groups", "mars_titan.data.asset_groups"} <= set(code)


def test_an_instant_split_into_blocks_is_refused(streams, tmp_path):
    engine = trainer(streams, tmp_path / "split", block_rows=2)
    with pytest.raises(ValueError, match="sesión completa"):
        engine.run()
