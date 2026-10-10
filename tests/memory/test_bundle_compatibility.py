"""Una sesión financiera solo admite una versión de edición, codec y receta.

Cada prueba prepara ediciones reales del corpus histórico con máscaras, sus índices de
observaciones, el predictor Titans-MAC congelado, el codec episódico y el ejecutor nativo,
y mezcla a propósito piezas de versiones distintas. Las ediciones de la prueba tienen las
mismas dimensiones, así que el rechazo depende de las identidades y no de las formas.
Ninguna prueba ejecuta pasos de optimizador: el predictor está en evaluación y sin
gradientes.
"""

import json

import pytest
import test_native_episode_backend as backend_fixtures
import torch
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_consumers import SOURCE_ERROR
from mars_titan.memory.financial_observations import (
    FinancialObservationSource,
    prepare_observation_index,
    run_observation_source,
)
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.titans.config import PAPER_PROJECTIONS
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.training.cohort_contract import representation_hash
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.prefix_eligibility import PrefixTargetVerifier
from tests.training.test_historical_corpus_inputs import supervised

pytestmark = pytest.mark.native_binding
native = backend_fixtures.native

PHASE = FinancialPhase(
    "train",
    946_684_800_000_000,
    946_684_800_000_000,
    1_672_531_200_000_000,
    1_672_531_200_000_000,
)
# El ejecutor nativo compara la identidad guardada (origen, vista, contrato y modelo).
OTHER_RUN = "otro origen, vista o modelo"


def edition(root, *, encoders=None):
    """Edición supervisada con su índice de la fase. `encoders` revisa la representación."""
    path = supervised(root)
    if encoders is not None:
        meta = json.loads(path.read_text())
        meta["representation"]["encoders"] = encoders
        digest = representation_hash(meta["representation"], input_policy=HISTORICAL_MASKED)
        for asset in meta["assets"]:
            asset["representation_sha256"] = digest
        path.write_text(json.dumps(meta))
    dataset = CorpusDataset(path, input_policy=HISTORICAL_MASKED)
    index = prepare_observation_index(dataset, root / "index", phase=PHASE)
    return dict(root=root, dataset=dataset, stream=FinancialObservationSource(dataset, index))


def bundle(data, *, readout_codec=None, **recipe):
    """Predictor, codec, prefijo y lectura episódica de una sola edición."""
    specification = data["stream"].specification()
    recipe = dict(variant="mac_online", hidden_size=32) | recipe
    predictor = FinancialPredictor(FinancialConfig(specification, **recipe), dtype=torch.float64)
    predictor.eval().requires_grad_(False)
    codec = FrozenEpisodeCodec(specification)
    readout = None
    if readout_codec is not None:
        readout = EpisodicReadout(
            EpisodicReadoutConfig(readout_codec.fingerprint(), hidden_size=recipe["hidden_size"]),
            dtype=torch.float64,
        )
        readout.eval().requires_grad_(False)
    return dict(
        specification=specification,
        consumer=FrozenFinancialConsumer(predictor, readout=readout),
        codec=codec,
        prefixes=PrefixTargetVerifier(
            data["dataset"], source_manifest=data["root"] / "materialized.json"
        ),
    )


def session(native, output, parts, *, resume=False, admission="m0"):
    return FinancialSession(
        output,
        native=native,
        consumer=parts["consumer"],
        codec=parts["codec"],
        prefixes=parts["prefixes"],
        retention=RetentionConfig(policy="reservoir", capacity=4, frontier=2, new_candidates=4),
        phase=PHASE,
        admission=admission,
        world="fixture",
        fold="0",
        resume=resume,
    )


def files(folder):
    """Bytes de todo lo que la sesión ha confirmado, para comprobar que nada cambia."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def editions(tmp_path):
    return edition(tmp_path / "a"), edition(tmp_path / "b")


def test_the_editions_differ_only_in_identity(editions):
    first, second = (data["stream"].specification() for data in editions)
    assert first.dimensions == second.dimensions
    assert first.representation == second.representation
    assert first.source_sha256 != second.source_sha256
    assert first.fingerprint() != second.fingerprint()


def test_one_consistent_bundle_runs_the_phase_and_resumes(native, editions, tmp_path):
    parts = bundle(editions[0])
    with session(native, tmp_path / "run", parts) as run:
        run_observation_source(editions[0]["stream"], run)
        final = run.snapshot()
    assert final["observed"] > 0 and final["issued"] > 0
    with session(native, tmp_path / "run", bundle(editions[0]), resume=True) as restored:
        assert restored.snapshot() == final


@pytest.mark.parametrize("mixed", ["codec", "prefixes"])
def test_a_codec_or_prefix_from_another_edition_is_rejected_before_any_output(
    native, editions, tmp_path, mixed
):
    parts = bundle(editions[0]) | {mixed: bundle(editions[1])[mixed]}
    with pytest.raises(ValueError, match=SOURCE_ERROR):
        session(native, tmp_path / "run", parts)
    assert not (tmp_path / "run").exists()


def test_a_readout_trained_for_another_codec_is_rejected(native, editions, tmp_path):
    other = bundle(editions[1])["codec"]
    parts = bundle(editions[0], readout_codec=other)
    with pytest.raises(ValueError, match=SOURCE_ERROR):
        session(native, tmp_path / "run", parts, admission="m1")
    assert not (tmp_path / "run").exists()


def test_a_materialized_edition_from_another_cohort_is_rejected(editions):
    first, second = editions
    with pytest.raises(ValueError, match="edición materializada original"):
        PrefixTargetVerifier(first["dataset"], source_manifest=second["root"] / "materialized.json")


def test_observations_of_another_edition_never_reach_the_session(native, editions, tmp_path):
    first, second = editions
    with session(native, tmp_path / "run", bundle(first)) as run:
        before = files(tmp_path / "run")
        with pytest.raises(ValueError, match="no está vinculada"):
            run_observation_source(second["stream"], run)
        # Un lote validado con la especificación de la otra edición tampoco entra a mano.
        event = next(event for event in second["stream"].events() if event.inputs)
        batch = validated_cpu_batch(event.inputs[0], second["stream"].specification())
        with pytest.raises(ValueError, match="pertenece a otro contrato"):
            run.step([batch], [], kind="warmup", cutoff=event.at)
        assert files(tmp_path / "run") == before
    # Dentro de la sesión lo detiene antes el codec. El predictor también lo rechaza solo.
    predictor = bundle(first)["consumer"].predictor
    device = DecisionBatch.from_validated(batch, dtype=torch.float64)
    with torch.no_grad(), pytest.raises(ValueError, match="no corresponde al predictor"):
        predictor.prepare(device, predictor.initial_state(device.flow_ids))


@pytest.fixture
def confirmed(native, editions, tmp_path):
    """Sesión de la edición A recorrida hasta el cierre de su fase."""
    with session(native, tmp_path / "run", bundle(editions[0])) as run:
        run_observation_source(editions[0]["stream"], run)
        final, contract = run.snapshot(), run.contract_id
    return final, files(tmp_path / "run"), contract


def test_resuming_with_another_edition_is_rejected_without_writing(
    native, editions, confirmed, tmp_path
):
    _, before, _ = confirmed
    with pytest.raises(ValueError, match=OTHER_RUN):
        session(native, tmp_path / "run", bundle(editions[1]), resume=True)
    assert files(tmp_path / "run") == before


@pytest.mark.parametrize(
    "recipe", [dict(hidden_size=64), dict(memory_projections=PAPER_PROJECTIONS)]
)
def test_resuming_with_another_recipe_is_rejected_without_writing(
    native, editions, confirmed, tmp_path, recipe
):
    _, before, _ = confirmed
    parts = bundle(editions[0], **recipe)
    with pytest.raises(ValueError, match=OTHER_RUN):
        session(native, tmp_path / "run", parts, resume=True)
    assert files(tmp_path / "run") == before


def test_resuming_with_another_admission_is_rejected_without_writing(
    native, editions, confirmed, tmp_path
):
    _, before, _ = confirmed
    with pytest.raises(ValueError, match=OTHER_RUN):
        session(native, tmp_path / "run", bundle(editions[0]), resume=True, admission="m1")
    assert files(tmp_path / "run") == before


@pytest.fixture
def revised(tmp_path):
    """Misma edición A con otra revisión de sus codificadores y las mismas dimensiones."""
    return edition(tmp_path / "r", encoders={"version": 2})


def test_a_cohort_cannot_mix_assets_encoded_with_another_representation(tmp_path):
    path = supervised(tmp_path / "mixed")
    meta = json.loads(path.read_text())
    changed = dict(meta["representation"], encoders={"version": 2})
    meta["assets"][0]["representation_sha256"] = representation_hash(
        changed, input_policy=HISTORICAL_MASKED
    )
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="identidad semántica común"):
        CorpusDataset(path, input_policy=HISTORICAL_MASKED)


def test_a_representation_change_keeps_the_shapes_but_not_the_identity(editions, revised):
    original = editions[0]["stream"].specification()
    changed = revised["stream"].specification()
    assert changed.dimensions == original.dimensions
    assert changed.representation != original.representation
    assert FrozenEpisodeCodec(changed).fingerprint() != FrozenEpisodeCodec(original).fingerprint()
    # El codec de la representación nueva no codifica observaciones de la anterior.
    event = next(event for event in editions[0]["stream"].events() if event.inputs)
    with pytest.raises(ValueError):
        FrozenEpisodeCodec(changed).encode(validated_cpu_batch(event.inputs[0], original))


def test_a_representation_change_requires_rebuilding_from_history(
    native, confirmed, revised, tmp_path
):
    final, before, original = confirmed
    # Los episodios, pesos rápidos y estados de A no se heredan en la representación nueva.
    with pytest.raises(ValueError, match=OTHER_RUN):
        session(native, tmp_path / "run", bundle(revised), resume=True)
    assert files(tmp_path / "run") == before
    # La reconstrucción vuelve a recorrer la historia permitida desde el inicio de la fase
    # en otra carpeta, sin tocar la sesión anterior.
    with session(native, tmp_path / "rebuilt", bundle(revised)) as run:
        run_observation_source(revised["stream"], run)
        rebuilt = run.snapshot()
        contract = run.contract_id
    assert files(tmp_path / "run") == before
    assert rebuilt["observed"] == final["observed"] and contract != original
