"""Lector episódico sobre un padre que necesita la sesión completa (atención entre activos)."""

import pytest
import torch

from mars_titan.models.titans.cross_asset_attention import CrossAssetAttentionConfig
from tests.training.test_financial_run import entries
from tests.training.test_mars_titan_run import (
    build,
    explicit_fastpath,  # noqa: F401
    native,  # noqa: F401
    recipe,
)
from tests.training.test_mars_titan_run_session_ic_term import (
    ASSETS,
    dispersed_parent,
    sessions,  # noqa: F401
)

ATTENTION = CrossAssetAttentionConfig(heads=4, mode="same_session")


def attentive_parent(sources):
    """Padre cuya atención mezcla los activos de la sesión, con la salida de la atención no nula."""
    parent = dispersed_parent(sources, cross_asset_attention=ATTENTION)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(29)
        weight = parent.cross_asset_attention.out.weight
        weight.copy_(torch.randn(weight.shape, generator=generator, dtype=weight.dtype) / 4)
    parent._seal_parameters()
    return parent


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_a_reader_runs_over_a_session_level_parent_when_each_session_fits_one_block(
    sessions,  # noqa: F811
    tmp_path,
    native,  # noqa: F811
    learning_doubles,
):
    parent = attentive_parent(sessions)
    assert parent.session_level
    engine = build(
        sessions,
        tmp_path / "completo",
        native,
        model=parent,
        plan=recipe(max_grad_norm=None, block_rows=ASSETS),
    )
    report = engine.run()
    assert report["status"] == "completed"
    emitted = entries(engine.audit, "prediction", "validation")
    assert len(emitted) > 0
    # La atención cuenta: sin ella, el mismo padre emite otra cosa en las mismas filas.
    plain = build(
        sessions,
        tmp_path / "sin-atencion",
        native,
        model=dispersed_parent(sessions),
        plan=recipe(max_grad_norm=None, block_rows=ASSETS),
    )
    plain.run()
    other = entries(plain.audit, "prediction", "validation")
    assert [row[2:4] for row in emitted] == [row[2:4] for row in other]
    assert any(first[4] != second[4] for first, second in zip(emitted, other, strict=True))


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_a_session_split_in_blocks_is_refused_for_a_session_level_parent(
    sessions,  # noqa: F811
    tmp_path,
    native,  # noqa: F811
    learning_doubles,
):
    engine = build(
        sessions,
        tmp_path / "partido",
        native,
        model=attentive_parent(sessions),
        plan=recipe(max_grad_norm=None, block_rows=2),
    )
    with pytest.raises(ValueError, match="sesión completa|sesión entera|session_complete"):
        engine.run()
