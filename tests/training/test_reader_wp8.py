"""WP8 (d) e integración de recetas/entrenador del lector en CPU con session_ic_term."""

import json
import os
from pathlib import Path

import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory import financial_observations as observations
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.mars_titan_variant import core_identity, load_declaration, select_variant
from mars_titan.memory.native_backend import load_native
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.session_ic_objective import SESSION_IC_TERM
from mars_titan.training import mars_titan_run as mt
from mars_titan.training import mars_titan_walk_forward
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.selection import FIXED_BUDGET
from tests.suite_support import missing_native
from tests.training.chronological_fixture import chronological_corpus, phases

ROOT = Path(__file__).resolve().parents[2]
CORE_RECIPE = ROOT / "configs/titans/dgx-us-core-u2-ic3-dense-gr-v9.json"
BASE_READER_RECIPE = ROOT / "configs/titans/dgx-us-reader-v2-s-v10.json"
NEW_READER_RECIPE = ROOT / "configs/titans/dgx-us-reader-v2-s-v10-ic3.json"
SELECTION = dict(metric="session_mae", patience=5, min_delta=0.0, stopping=FIXED_BUDGET)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def cpu_sessions(tmp_path_factory):
    root = tmp_path_factory.mktemp("wp8-reader-sessions")
    dataset = CorpusDataset(
        chronological_corpus(root / "corpus", assets=5), input_policy=HISTORICAL_MASKED
    )
    result = {}
    for phase in phases():
        manifest = observations.prepare_observation_index(
            dataset, root / f"index-{phase.partition}", phase=phase
        )
        result[phase.partition] = observations.FinancialObservationSource(dataset, manifest)
    return result


def dispersed_parent(sessions):
    config = FinancialConfig(
        sessions["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        seed=42,
        head="quantile_head_v1",
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
    )
    return FinancialPredictor(config, dtype=torch.float64).eval().requires_grad_(False)


def test_new_reader_recipe_copies_base_and_declares_parent_session_ic_term():
    base = mt.load_recipe(BASE_READER_RECIPE)
    derived = mt.load_recipe(NEW_READER_RECIPE)
    core = json.loads(CORE_RECIPE.read_text(encoding="utf-8"))

    assert derived["schema_version"] == base["schema_version"] == 1
    assert derived["recipe_name"] == base["recipe_name"]
    assert derived["status"] == "declared_not_executed"
    assert derived["walk_forward"] == base["walk_forward"]
    assert "session_ic_term" not in base["recipe"]
    assert derived["recipe"]["session_ic_term"] == core["recipe"]["session_ic_term"]
    assert {k: v for k, v in derived["recipe"].items() if k != "session_ic_term"} == base["recipe"]

    for search_case in derived["walk_forward"]["search_cases"]:
        plan = mt.case_recipe(derived, search_case)
        term = plan.session_ic()
        assert term is not None
        assert term.weight == 3.0 and term.minimum_rows == 8
        assert plan.identity()["session_ic_term"]["name"] == SESSION_IC_TERM


def test_readout_recipe_and_window_propagate_wp8_options_without_changing_default_identity():
    plain = mt.ReadoutRecipe()
    assert plain.learned_temperature is False
    assert plain.label_path == "concatenated"
    assert plain.label_scale == 1.0
    plain_id = plain.identity()
    for hidden in ("learned_temperature", "label_path", "label_scale"):
        assert hidden not in plain_id

    wide = mt.ReadoutRecipe(
        neighbors=64,
        temperature=0.1,
        learned_temperature=True,
        label_path="direct_v1",
        label_scale=25.0,
    )
    wide_id = wide.identity()
    assert wide_id["neighbors"] == 64
    assert wide_id["temperature"] == 0.1
    assert wide_id["learned_temperature"] is True
    assert wide_id["label_path"] == "direct_v1"
    assert wide_id["label_scale"] == 25.0

    for bad in (
        dict(neighbors=257),
        dict(learned_temperature=True, temperature=1.0),
        dict(learned_temperature=1),
        dict(label_path="unknown"),
        dict(label_path="concatenated", label_scale=5.0),
        dict(label_path="direct_v1", label_scale=0.0),
    ):
        with pytest.raises(ValueError):
            mt.ReadoutRecipe(**bad)


def test_window_builder_passes_wp8_options_to_readout(cpu_sessions):
    predictor = dispersed_parent(cpu_sessions)
    codec = FrozenEpisodeCodec(cpu_sessions["train"].specification())
    base = core_identity(predictor, dict(recipe_sha256="0" * 64, search_case=None, window="w"))
    variant = select_variant(load_declaration(), {"episodic_bank": "m1"}, base=base)
    plan = mt.ReadoutRecipe(
        neighbors=32,
        temperature=0.15,
        learned_temperature=True,
        label_path="direct_v1",
        label_scale=10.0,
    )
    readout = mars_titan_walk_forward._readout(variant, codec, predictor, plan, 42)
    assert readout.config.neighbors == 32
    assert readout.config.temperature == 0.15
    assert readout.config.learned_temperature is True
    assert readout.config.label_path == "direct_v1"
    assert readout.config.label_scale == 10.0


def test_reader_trains_at_least_one_step_on_cpu_with_new_recipe_session_ic_term(
    cpu_sessions, tmp_path
):
    # Cargamos la receta nueva y ajustamos un paso real de ReadoutTrainer en CPU (M0 no requiere
    # binario nativo) con su término session_ic_term y contra su control sin término de IC.
    document = mt.load_recipe(NEW_READER_RECIPE)
    base_plan = mt.case_recipe(document, "lr1e-3")
    term = dict(document["recipe"]["session_ic_term"], minimum_rows=3)

    def run_once(name, session_ic_term):
        plan = mt.ReadoutRecipe(
            update_instants=3,
            loss=base_plan.loss,
            learning_rate=base_plan.learning_rate,
            weight_decay=base_plan.weight_decay,
            max_grad_norm=base_plan.max_grad_norm,
            epochs=1,
            selection=SELECTION,
            block_rows=8,
            neighbors=base_plan.neighbors,
            temperature=base_plan.temperature,
            bank_capacity=8,
            bank_seed=base_plan.bank_seed,
            max_working_bytes=128 * 1024**2,
            checkpoint_updates=256,
            checkpoint_seconds=900.0,
            session_ic_term=session_ic_term,
        )
        predictor = dispersed_parent(cpu_sessions)
        codec = FrozenEpisodeCodec(cpu_sessions["train"].specification())
        readout = EpisodicReadout(
            EpisodicReadoutConfig(
                codec.fingerprint(),
                hidden_size=32,
                mode="no_bank",
                neighbors=plan.neighbors,
                temperature=plan.temperature,
                max_working_bytes=plan.max_working_bytes,
                seed=42,
            ),
            dtype=torch.float64,
        )
        snapshots = {}

        def capture(event, modules):
            snapshots[event["epoch"]] = {
                key: value.detach().clone() for key, value in modules["readout"].named_parameters()
            }

        trainer = mt.ReadoutTrainer(
            predictor,
            readout,
            plan,
            admission="m0",
            retention=None,
            native=None,
            codec=codec,
            train=cpu_sessions["train"],
            validation=cpu_sessions["validation"],
            output=tmp_path / name,
            world="wp8",
            fold="0",
        )
        trainer.trace = capture
        report = trainer.run()
        return trainer, report, snapshots

    trainer_ic, report_ic, snaps_ic = run_once("reader-ic3-m0", term)
    assert trainer_ic.session_ic is not None
    assert trainer_ic.identity["recipe"]["session_ic_term"]["name"] == SESSION_IC_TERM
    assert report_ic["status"] == "completed"
    assert trainer_ic.global_step >= 1
    assert any(
        int(state["step"].item()) >= 1
        for state in trainer_ic.optimizer.state.values()
        if "step" in state
    )
    # Aunque la selección por validación pueda recuperar la época 0 como mejor punto, la época 1
    # entrenada debe diferir de la inicialización tras el paso real de optimizador.
    assert 0 in snaps_ic and 1 in snaps_ic
    assert any(not torch.equal(snaps_ic[1][k], snaps_ic[0][k]) for k in snaps_ic[0])

    # Además, activar session_ic_term cambia efectivamente el paso frente a la misma receta sin IC.
    _, report_plain, snaps_plain = run_once("reader-plain-m0", None)
    assert report_plain["status"] == "completed"
    assert all(torch.equal(snaps_ic[0][k], snaps_plain[0][k]) for k in snaps_ic[0])
    assert any(not torch.equal(snaps_ic[1][k], snaps_plain[1][k]) for k in snaps_ic[1])


def test_parameter_roles_and_inference_contract_cover_wp8_options(cpu_sessions):
    predictor = dispersed_parent(cpu_sessions)
    codec = FrozenEpisodeCodec(cpu_sessions["train"].specification())
    plan = mt.ReadoutRecipe(
        neighbors=16,
        temperature=0.2,
        learned_temperature=True,
        label_path="direct_v1",
        label_scale=10.0,
        block_rows=8,
        bank_capacity=16,
        max_working_bytes=128 * 1024**2,
    )
    readout = EpisodicReadout(
        EpisodicReadoutConfig(
            codec.fingerprint(),
            hidden_size=32,
            mode="bank",
            neighbors=plan.neighbors,
            temperature=plan.temperature,
            learned_temperature=plan.learned_temperature,
            label_path=plan.label_path,
            label_scale=plan.label_scale,
            max_working_bytes=plan.max_working_bytes,
            seed=42,
        ),
        dtype=torch.float64,
    )
    roles, inert = mt.parameter_roles(readout, "m1")
    assert inert == []
    assert "temperature_logit" in roles["episodic_read"]
    assert "label_gain" in roles["episodic_read"]
    assert "temperature_logit" not in roles["refiner"]
    assert "label_gain" not in roles["refiner"]

    # Con coincidencia exacta, MarsTitanInference admite la configuración WP8 (incluso con stub
    # nativo para comprobar el contrato sin compilar el binario).
    from types import SimpleNamespace

    stub_native = SimpleNamespace(MemoryRecord=object)
    mt.MarsTitanInference(
        predictor,
        readout,
        plan,
        admission="m1",
        retention=mt.retention_config(plan, "m1"),
        native=stub_native,
        codec=codec,
        world="wp8",
        fold="0",
    )

    # MarsTitanInference rechaza discrepancias en las opciones WP8 entre receta y lector.
    for mismatch in (
        dict(learned_temperature=False),
        dict(label_path="concatenated", label_scale=1.0),
        dict(label_scale=20.0),
    ):
        wrong_plan = mt.ReadoutRecipe(
            neighbors=plan.neighbors,
            temperature=plan.temperature,
            learned_temperature=mismatch.get("learned_temperature", plan.learned_temperature),
            label_path=mismatch.get("label_path", plan.label_path),
            label_scale=mismatch.get("label_scale", plan.label_scale),
            block_rows=plan.block_rows,
            bank_capacity=plan.bank_capacity,
            max_working_bytes=plan.max_working_bytes,
        )
        with pytest.raises(ValueError, match="no corresponde"):
            mt.MarsTitanInference(
                predictor,
                readout,
                wrong_plan,
                admission="m1",
                retention=mt.retention_config(wrong_plan, "m1"),
                native=stub_native,
                codec=codec,
                world="wp8",
                fold="0",
            )

    # También con label_scale=1.0 compartido, discrepar solo en label_path se rechaza.
    direct_unit_readout = EpisodicReadout(
        EpisodicReadoutConfig(
            codec.fingerprint(),
            hidden_size=32,
            mode="bank",
            neighbors=plan.neighbors,
            temperature=plan.temperature,
            label_path="direct_v1",
            label_scale=1.0,
            max_working_bytes=plan.max_working_bytes,
            seed=42,
        ),
        dtype=torch.float64,
    )
    concat_unit_plan = mt.ReadoutRecipe(
        neighbors=plan.neighbors,
        temperature=plan.temperature,
        label_path="concatenated",
        label_scale=1.0,
        block_rows=plan.block_rows,
        bank_capacity=plan.bank_capacity,
        max_working_bytes=plan.max_working_bytes,
    )
    with pytest.raises(ValueError, match="no corresponde"):
        mt.MarsTitanInference(
            predictor,
            direct_unit_readout,
            concat_unit_plan,
            admission="m1",
            retention=mt.retention_config(concat_unit_plan, "m1"),
            native=stub_native,
            codec=codec,
            world="wp8",
            fold="0",
        )


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_banked_reader_trains_on_cpu_with_wp8_options_and_session_ic_term(cpu_sessions, tmp_path):
    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not path:
        pytest.skip(missing_native("MARS_TITAN_EPISODIC_NATIVE"))
    native = load_native(path)
    document = mt.load_recipe(NEW_READER_RECIPE)
    term = dict(document["recipe"]["session_ic_term"], minimum_rows=3)
    plan = mt.ReadoutRecipe(
        update_instants=3,
        epochs=1,
        selection=SELECTION,
        block_rows=8,
        neighbors=16,
        temperature=0.2,
        learned_temperature=True,
        label_path="direct_v1",
        label_scale=10.0,
        refinement_initialization="zero_refinement",
        bank_capacity=16,
        max_working_bytes=128 * 1024**2,
        session_ic_term=term,
    )
    predictor = dispersed_parent(cpu_sessions)
    codec = FrozenEpisodeCodec(cpu_sessions["train"].specification())
    readout = EpisodicReadout(
        EpisodicReadoutConfig(
            codec.fingerprint(),
            hidden_size=32,
            mode="bank",
            neighbors=plan.neighbors,
            temperature=plan.temperature,
            learned_temperature=plan.learned_temperature,
            label_path=plan.label_path,
            label_scale=plan.label_scale,
            refinement_initialization=plan.refinement_initialization,
            max_working_bytes=plan.max_working_bytes,
            seed=42,
        ),
        dtype=torch.float64,
    )
    trainer = mt.ReadoutTrainer(
        predictor,
        readout,
        plan,
        admission="m1",
        retention=mt.retention_config(plan, "m1"),
        native=native,
        codec=codec,
        train=cpu_sessions["train"],
        validation=cpu_sessions["validation"],
        output=tmp_path / "reader-ic3-m1",
        world="wp8",
        fold="0",
    )
    assert "label_gain" in trainer.roles["episodic_read"]
    assert "temperature_logit" in trainer.roles["episodic_read"]
    report = trainer.run()
    assert report["status"] == "completed"
    assert trainer.global_step >= 1
    assert torch.isfinite(readout.label_gain).all()
    assert 0.02 <= readout.effective_temperature().item() <= 1.0
