"""Declaraciones estadísticas de fixtures para el estudio E2, sin ejecuciones reales."""

import copy

ANALYSIS = dict(
    block_length=16,
    replicates=2000,
    seed=20261027,
    confidence=0.95,
    market_weighting="session",
    sensitivity_block_lengths=[8, 32],
    rank_ic_min_assets=3,
    primary_contrast="utility_minus_baseline",
    min_relevant_mae=0.00001,
)


def analysis_document(document, reader_contract):
    return dict(
        copy.deepcopy(document),
        schema_version=3,
        reader_contract=reader_contract,
        analysis=copy.deepcopy(ANALYSIS),
    )
