"""Inventario del bloqueo de aprendizaje: cada guarda, la prueba que la demuestra y las exenciones.

`GUARDS` asigna a cada función que llama a `require_learning_allowed` una referencia por
llamada, en el orden del código. Una referencia sin `::` es una clave de `ENTRY_POINTS` en
`test_learning_hold_guards.py`, cuya prueba comprueba además que la detiene justo esa
guarda. Las demás nombran una prueba de otro módulo como `ruta::función`.

`test_learning_hold_inventory.py` exige que el inventario coincida con las guardas que
encuentra el grafo de llamadas, así que una guarda nueva o retirada obliga a revisarlo.
"""


def node(path, test):
    """Prueba de otro módulo, con la ruta relativa a `tests`."""
    return f"tests/{path}::{test}"


_POSTTRAINING_ENTRIES = node(
    "posttraining/test_learning_hold_entrypoints.py",
    "test_entry_points_refuse_before_opening_sources_or_outputs",
)
_ADAPTER_STAGE = node(
    "posttraining/test_adapter_campaign_stage.py",
    "test_the_hold_blocks_before_reading_and_before_each_pending_job",
)
_CANDIDATE_WINDOWS = node(
    "training/test_candidate_walk_forward.py", "test_window_entries_need_the_learning_permission"
)
_REGENERATION = node(
    "training/test_prediction_regeneration.py",
    "test_regeneration_stops_while_the_learning_hold_blocks",
)

GUARDS = {
    # Sondas, perfilado y comparadores tabulares.
    "mars_titan.budget_training:train_budget_grid": ("budget_training",),
    "mars_titan.budget_training:train_step": ("budget_train_step",),
    "mars_titan.gru_probe:run_temporal_probe": ("gru_probe",),
    "mars_titan.profiling:profile_case": ("profiling",),
    "mars_titan.reference_probe:run_reference_probe": ("reference_probe",),
    "mars_titan.models.baselines.boosting:fit_boosting_batches": ("fit_boosting_batches",),
    "mars_titan.models.baselines.campaign:run_campaign": ("baseline_campaign",),
    "mars_titan.models.baselines.external_boosting:fit_external_boosting": (
        "fit_external_boosting",
    ),
    "mars_titan.models.baselines.ridge:fit_ridge_blocks": ("fit_ridge_blocks",),
    "mars_titan.models.baselines.ridge:solve_ridge": ("solve_ridge",),
    # Corpus, búsquedas y campañas de referencia.
    "mars_titan.training.reference_run:run_reference_case": ("reference_run",),
    "mars_titan.training.reference_search:run_search": ("reference_search",),
    "mars_titan.training.temporal_search:run_temporal_search": ("temporal_search",),
    "mars_titan.training.reference_campaign:run_reference_campaign": ("reference_campaign",),
    "mars_titan.training.tabular_corpus:run_tabular_reference": ("tabular_corpus",),
    "mars_titan.training.tabular_search:run_tabular_search": ("tabular_search",),
    "mars_titan.training.external_corpus:run_external_reference": ("external_corpus",),
    "mars_titan.training.external_corpus:_execute": (
        node(
            "training/test_external_corpus.py",
            "test_the_executor_stops_under_the_hold_before_touching_its_output",
        ),
    ),
    "mars_titan.training.predictive_run:run_predictive_case": ("predictive_run",),
    "mars_titan.training.predictive_study:run_predictive_study": ("predictive_study",),
    "mars_titan.training.baseline_queue:run_queue": ("baseline_queue",),
    "mars_titan.training.real_campaign:run_campaign": ("real_campaign",),
    "mars_titan.training.online_reference:run_online_reference": ("online_reference",),
    "mars_titan.training.financial_run:ChronologicalTrainer.run": (
        node(
            "training/test_financial_run.py",
            "test_real_optimizer_is_refused_while_the_learning_hold_blocks",
        ),
    ),
    "mars_titan.training.quantile_head_control:run_control": (
        node(
            "training/test_quantile_head_control.py",
            "test_run_is_blocked_before_creating_any_output",
        ),
        node(
            "training/test_quantile_head_control.py",
            "test_hold_reinstated_after_a_job_stops_before_the_next_one",
        ),
    ),
    # Candidato GRU episódico.
    "mars_titan.training.candidate_run:CandidateChronologicalTrainer.run": (
        node(
            "training/test_candidate_run.py",
            "test_active_hold_stops_the_trainer_before_creating_outputs",
        ),
    ),
    "mars_titan.training.candidate_walk_forward:fit_window": (_CANDIDATE_WINDOWS,),
    "mars_titan.training.candidate_walk_forward:carry_window": (_CANDIDATE_WINDOWS,),
    # Titans-MAC, MARS-TITAN, B6 y CM-v1.
    "mars_titan.training.titans_walk_forward:run_titans_window": ("titans_walk_forward",),
    "mars_titan.training.titans_walk_forward:carry_titans": ("carry_titans",),
    "mars_titan.training.mars_titan_run:ReadoutTrainer.run": (
        node(
            "training/test_mars_titan_run.py",
            "test_readout_fit_is_refused_while_the_learning_hold_blocks",
        ),
    ),
    "mars_titan.training.mars_titan_walk_forward:run_mars_titan_window": (
        "mars_titan_walk_forward",
    ),
    "mars_titan.training.mars_titan_walk_forward:run_readout_window": ("readout_window",),
    "mars_titan.training.mars_titan_walk_forward:carry_mars_titan": ("carry_mars_titan",),
    "mars_titan.training.mars_titan_walk_forward:carry_readout": ("carry_readout",),
    "mars_titan.training.mars_titan_correction:run_correction_window": (
        node(
            "training/test_mars_titan_correction.py",
            "test_window_refuses_while_the_learning_hold_blocks",
        ),
    ),
    "mars_titan.training.mars_titan_correction:carry_correction": ("carry_correction",),
    "mars_titan.training.cm_v1_factorial:run_cm_v1_core_window": ("cm_v1_core_window",),
    "mars_titan.training.cm_v1_factorial:run_cm_v1_window": ("cm_v1_window",),
    "mars_titan.training.cm_v1_factorial:carry_cm_v1": ("carry_cm_v1",),
    # Campaña con máscaras y regeneración de predicciones.
    "mars_titan.training.masked_campaign:run_campaign": (
        node(
            "training/test_masked_campaign.py",
            "test_active_hold_rejects_the_campaign_before_creating_outputs",
        ),
    ),
    "mars_titan.training.masked_campaign:_execute": (
        node(
            "training/test_masked_campaign.py",
            "test_hold_reinstated_during_the_campaign_stops_before_the_next_job",
        ),
    ),
    "mars_titan.training.prediction_regeneration:regenerate_job": (_REGENERATION,),
    "mars_titan.training.prediction_regeneration:regenerate_ablation": (_REGENERATION,),
    # Posentrenamiento y adaptadores.
    "mars_titan.posttraining.run:run_case": (_POSTTRAINING_ENTRIES,),
    "mars_titan.posttraining.queue:run_queue": (_POSTTRAINING_ENTRIES,),
    "mars_titan.posttraining.queue:run_matrix_queue": (
        node(
            "posttraining/test_matrix_queue.py",
            "test_matrix_mode_stops_before_reading_with_the_hold",
        ),
    ),
    "mars_titan.posttraining.completion:_stage": (_POSTTRAINING_ENTRIES,),
    "mars_titan.posttraining.completion:run_completion": (_POSTTRAINING_ENTRIES,),
    "mars_titan.posttraining.campaign_stage:run_stage": (_ADAPTER_STAGE,),
    "mars_titan.posttraining.campaign_stage:_Stage.execute": (_ADAPTER_STAGE,),
    "mars_titan.posttraining.candidate_adapters:run_candidate_posttraining": (
        "candidate_posttraining",
    ),
    "mars_titan.posttraining.candidate_adapters:frozen_candidate": ("frozen_candidate",),
    "mars_titan.posttraining.chronological_windows:run_titans_posttraining": (
        "titans_posttraining",
    ),
    "mars_titan.posttraining.chronological_windows:run_readout_posttraining": (
        "readout_posttraining",
    ),
    "mars_titan.posttraining.chronological_windows:frozen_titans": ("frozen_titans",),
    "mars_titan.posttraining.chronological_windows:frozen_readout": ("frozen_readout",),
    # Simulación y políticas.
    "mars_titan.simulation.training:FinancialTrainer.run": ("financial_trainer",),
    "mars_titan.simulation.campaign:run_campaign": ("financial_comparators",),
    "mars_titan.simulation.adaptive_campaign:run_adaptive_campaign": ("adaptive_campaign",),
    "mars_titan.simulation.adaptation_scenarios:prepare_adaptation_scenarios": (
        "adaptation_scenarios_hmm",
    ),
    "mars_titan.simulation.adaptation_scenarios:fit_hmm": ("fit_hmm",),
    "mars_titan.training.klpo_queue:run_queue": ("klpo_queue",),
    "mars_titan.simulation.campaign_stage:run_stage": (
        node(
            "simulation/test_campaign_stage.py",
            "test_the_hold_stops_the_stage_before_probing_or_launching_binaries",
        ),
    ),
    "mars_titan.simulation.campaign_stage:_Stage.execute": (
        node(
            "simulation/test_campaign_stage.py",
            "test_the_learning_hold_stops_the_stage_before_outputs_and_between_jobs",
        ),
    ),
    # Lanzadores. La auditoría congelada solo se bloquea sobre cintas reconstruidas.
    "scripts.run_native_ppo:main": ("run_native_ppo", "run_native_ppo_reconstructed_audit"),
    "scripts.benchmark_native_ppo:main": ("benchmark_native_ppo",),
    "scripts.benchmark_adaptive_rl:main": ("benchmark_adaptive_rl",),
    "scripts.run_financial_comparators:main": ("run_financial_comparators",),
    "scripts.run_titans_walk_forward:main": ("run_titans_walk_forward",),
}

# Funciones que registran un gancho de PyTorch que rechaza cualquier paso mientras miden. Solo
# cubren los pasos de torch.optim que se alcanzan desde ellas, no un estimador ni XGBoost.
NO_STEP_HOOKS = {
    "mars_titan.training.campaign_throughput:_forbid_steps": (
        "La medición de caudal recorre los entrenadores reales con un optimizador sin pasos"
    ),
    "mars_titan.training.modality_ablation_stage:forbid_optimizer_steps": (
        "La ablación de modalidades predice con padres congelados durante toda la etapa"
    ),
}

# Sitios que ajustan en sentido amplio pero forman parte de la predicción en línea.
EXEMPT_SITES = {
    "mars_titan.memory.associative_memory:AssociativeMemory._kalman": (
        "La memoria asociativa de B6 escribe en su estado los resultados ya maduros durante la "
        "inferencia. No cambia parámetros compartidos y sus reglas delta y proximal hacen la "
        "misma escritura sin resolver un sistema"
    ),
}
