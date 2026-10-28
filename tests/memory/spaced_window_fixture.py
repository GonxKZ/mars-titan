"""Ventana histórica técnica con objetivos manuales y pesos fijados, sin aprendizaje."""

import json
from dataclasses import asdict
from pathlib import Path

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.splits import build_folds
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.prefix_eligibility import PrefixTargetVerifier
from mars_titan.training.temporal_corpus import prepare_temporal_corpus
from mars_titan.training.titans_walk_forward import _sources, window_phases
from tests.memory.spaced_selected_fixture import selected_fixture
from tests.training.historical_temporal_fixture import DEFAULT_DAYS, historical_temporal_fixture

PROTOCOL = dict(
    schema_version=2,
    market="US",
    train_start="2000-01-01",
    first_validation_start="2023-09-01",
    validation_months=1,
    calibration_months=1,
    evaluation_months=2,
    step_months=2,
    minimum_train_months=240,
    purge="label_interval",
    final_test_start="2024-01-01",
    final_test_end="2025-01-01",
    primary_metric="session_mae",
    selection=dict(
        metric="session_mae", stopping="fixed_budget", patience=1, min_delta=0.00001, max_epochs=2
    ),
    seeds=[42],
)


def study_fixture(
    root,
    *,
    assets=2,
    rich=False,
    components=None,
    reader_contract="standard",
    price_row_group_size=3,
):
    root = Path(root)
    days = None
    if rich:
        days = dict(US=sorted({*DEFAULT_DAYS, "2023-11-15", "2023-12-01", "2023-12-15"}))
    data = historical_temporal_fixture(
        root / "data", assets=assets, days=days, price_row_group_size=price_row_group_size
    )
    metadata = json.loads(data.parent.read_text())
    first = metadata["assets"][0]
    factor = Path(metadata["roots"]["prepared"]) / "US" / first["symbol"] / "prices.parquet"
    factors = dict(
        US=dict(
            market="US",
            symbol="FIXTURE_FACTOR",
            prices_path=str(factor),
            prices_sha256=sha256(factor),
        )
    )
    original = root / "original-source.json"
    atomic_json(
        original,
        dict(
            kind="manual_source_fixture",
            calendar_start=dict(US="1999-01-01"),
            market_factors=factors,
        ),
    )
    metadata.update(
        market_factors=factors, configuration=dict(source_manifest_sha256=sha256(original))
    )
    atomic_json(data.parent, metadata)
    protocol_path = root / "protocol.json"
    atomic_json(protocol_path, PROTOCOL)
    prepare_temporal_corpus(
        data.parent,
        protocol_path,
        None,
        None,
        root / "views",
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    view = root / "views/fold-000/manifest.json"
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    fold = build_folds(PROTOCOL)[0]
    phases = window_phases(fold, 0)
    sources = _sources(dataset, phases, root / "indices")
    spec = sources["train"].specification()
    source = dict(
        spec=spec,
        codec=FrozenEpisodeCodec(spec),
        prefixes=PrefixTargetVerifier(dataset, source_manifest=original),
        window=fold,
        phases={name: asdict(phase) for name, phase in phases.items()},
        bank_capacity=min(1024, max(16, assets * 4)),
    )
    if components is not None:
        source["components"] = components
    selected = selected_fixture(root / "selected", source)
    path = (
        "configs/experiments/spaced-regional-utility-study-v1.json"
        if reader_contract == "regional"
        else "configs/experiments/spaced-utility-study-v1.json"
    )
    config = json.loads(Path(path).read_text())
    config["retention"].update(capacity=source["bank_capacity"], frontier=1)
    config["compute_budget"]["max_phase_rows"] = max(100, assets * 10)
    study = root / "study.json"
    atomic_json(study, config)
    return dict(
        selected,
        view=view,
        source_manifest=original,
        dataset=dataset,
        study=study,
        indices=root / "indices",
        phases=phases,
    )
