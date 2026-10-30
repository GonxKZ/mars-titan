"""Pruebas que rebajan la protección del aprendizaje, con su motivo (#546).

Mientras rige el bloqueo de aprendizaje ninguna prueba puede rebajarlo fuera de lo declarado
aquí. `tests/test_learning_hold_guard.py` recorre las pruebas, las medidas y los scripts y
falla ante cualquier rebaja nueva.

`INVENTORY` reúne las pruebas que necesitan la protección abierta de verdad: ajustes nativos
que crean un entrenador y su Adam, el binario de entrenamiento que pasa una protección
permitida o comprobaciones de que la protección se vuelve a leer entre trabajos, que exigen
empezar con ella abierta. Se marcan con `lowers_learning_hold` y
se omiten mientras la protección real bloquee, así que no la rebajan durante el bloqueo.
Las que solo leen cintas o evalúan sin aprender usan los modos `--check-sources` e
`--initial-evaluation` de los binarios, que la protección admite sin rebajarse.

`allowed_hold` es el único punto de las pruebas que escribe una protección permitida. Lo usan
el fixture `learning_doubles` de `tests/conftest.py` y los fixtures de módulo que recorren
campañas con el aprendizaje sustituido por dobles.
"""

import json
from pathlib import Path

import pytest

from mars_titan.training.learning_hold import learning_blocked

_POLICY_TAPES = "tests/simulation/test_native_policy_tapes.py::"
_NATIVE_TRACES = "tests/learning_traces/test_native_policy_traces.py::"

# Motivos compartidos por varias pruebas de las opciones de la campaña A v2 (#363).
_LAUNCH = (
    "la ventana reducida se lanza con el motor y los ejecutores reales hasta la entrada de "
    "cada entrenador, donde los dobles la detienen sin ajustar, y el motor exige la protección "
    "antes de leer la campaña"
)
_ADAPTER = (
    "`run_case` exige la protección antes de leer el caso, y el optimizador sustituido de "
    "`recorder` registra los gradientes sin cambiar pesos"
)

# Pruebas CUDA del ejecutor de ventanas que completan actualizaciones reales sobre la vista
# corta del fixture con la protección permitida de `allowed_child` (#745).
_WINDOWS = "tests/training/test_window_variants.py::"
_WINDOW_UPDATES = (
    "ejecuta en CUDA el ejecutor de variantes por ventana sobre la vista corta del fixture con "
    "la protección permitida de `allowed_child` y actualiza pesos de verdad"
)

INVENTORY = {
    "tests/training/test_campaign_a_joint.py::"
    "test_integrity_recounts_the_rows_the_joint_design_drops": (
        "publica la comparación conjunta desde la campaña A v2 reducida del fixture "
        "`joint_campaign`, que ejecuta los dobles de ajuste con la protección rebajada"
    ),
    "tests/training/test_campaign_a_joint.py::"
    "test_integrity_catches_a_comparison_that_drops_other_rows": (
        "publica la comparación conjunta desde la campaña A v2 reducida del fixture "
        "`joint_campaign`, que ejecuta los dobles de ajuste con la protección rebajada"
    ),
    "tests/integrity/test_fit_budget.py::"
    "test_a_chronological_fit_with_unchanged_weights_keeps_its_initial_state": (
        "lee los informes reales de las cuatro variantes de Titans-MAC del fixture `base`, "
        "que las ejecuta con la protección rebajada y un optimizador que no cambia pesos"
    ),
    "tests/integrity/test_fit_budget.py::test_a_neural_reference_from_scratch_counts_as_fitted": (
        "ejecuta la referencia neuronal enmascarada en CPU con la protección rebajada por "
        "`cpu_runner` para leer su informe real"
    ),
    _POLICY_TAPES + "test_ppo_pauses_on_real_tapes_before_its_first_update": (
        "lanza el ajuste PPO nativo, que crea el entrenador y su Adam, y lo pausa antes del "
        "primer recorrido para comprobar la pausa, la reanudación y que una selección abierta "
        "no se evalúa"
    ),
    _POLICY_TAPES + "test_double_dqn_selects_and_evaluates_the_chosen_state_without_updates": (
        "ajusta Double DQN con 32 transiciones, por debajo de su calentamiento, para recorrer "
        "la selección cerrada y su evaluación con --audit-run"
    ),
    _POLICY_TAPES + "test_group_objectives_publish_their_groups_before_any_update": (
        "lanza el ajuste KLPO de cada objetivo de grupo, que crea su Adam, y lo pausa con "
        "--stop-after al tener lista la primera oleada, para comprobar que publica sus grupos "
        "sin ningún paso de optimizador y los repite al reanudar. Supera las 32 transiciones "
        "del diagnóstico en CPU, así que necesita cuda:0"
    ),
    _POLICY_TAPES + "test_klpo_collects_a_full_wave_and_pauses_before_update_ready": (
        "recoge una oleada KLPO con el controlador y su Adam terminal y fabrica una selección "
        "cerrada para recorrer --audit-run"
    ),
    _NATIVE_TRACES + "test_double_dqn_traces_keep_the_run_and_record_each_validation": (
        "ajusta Double DQN con 32 transiciones, por debajo de su calentamiento, con y sin trazas "
        "para comprobar que las trazas no cambian la ejecución"
    ),
    _NATIVE_TRACES + "test_ppo_traces_are_not_duplicated_by_a_pause_and_resume": (
        "lanza el ajuste PPO nativo, que crea el entrenador y su Adam, y lo pausa y reanuda antes "
        "del primer recorrido para comprobar que las trazas no se duplican"
    ),
    _NATIVE_TRACES + "test_klpo_traces_record_the_initial_validation_once": (
        "lanza el ajuste KLPO nativo con el controlador y su Adam terminal y lo pausa antes de "
        "update_ready para comprobar que la validación inicial se traza una vez"
    ),
    _POLICY_TAPES + "test_stage_executors_fit_carry_and_pause_through_the_launcher": (
        "los ejecutores de la etapa lanzan ajustes nativos de Double DQN y KLPO"
    ),
    _POLICY_TAPES + "test_the_stage_confirms_the_declared_costs_of_a_fit": (
        "la etapa confirma los costes de un ajuste nativo de Double DQN lanzado por su ejecutor"
    ),
    "tests/simulation/test_native_learning_hold.py::"
    "test_training_passes_an_allowed_hold_and_fails_on_missing_sources": (
        "lanza el entrenamiento PPO nativo con la protección permitida para comprobar que pasa la "
        "puerta y se detiene al no encontrar las cintas, sin construir el entorno"
    ),
    "tests/simulation/test_campaign_stage.py::test_the_learning_hold_is_read_again_between_jobs": (
        "comprueba que la etapa de políticas vuelve a leer la protección entre trabajos, lo que "
        "exige empezar con ella abierta"
    ),
    "tests/training/test_masked_campaign.py::"
    "test_hold_reinstated_during_the_campaign_stops_before_the_next_job": (
        "comprueba que la campaña con máscaras vuelve a leer la protección entre trabajos"
    ),
    "tests/training/test_modality_ablation_stage.py::"
    "test_the_hold_is_checked_again_before_each_pending_job": (
        "comprueba que la etapa de ablación vuelve a leer la protección entre trabajos"
    ),
    "tests/posttraining/test_adapter_campaign_stage.py::"
    "test_the_hold_is_read_again_before_each_pending_job": (
        "comprueba que la etapa de adaptadores vuelve a leer la protección entre trabajos"
    ),
    "tests/training/test_quantile_head_control.py::"
    "test_hold_reinstated_after_a_job_stops_before_the_next_one": (
        "el primer trabajo del control de la cabeza tiene que pasar la protección para "
        "comprobar que vuelve a leerse antes del segundo, aunque el ejecutor sustituido no ajusta"
    ),
    "tests/training/test_modality_ablation_carry.py::"
    "test_b6_ablation_equals_the_correction_with_a_real_absence": (
        "prepara dos ventanas B6 con run_correction_window y traslada una con carry_correction "
        "sobre ventanas técnicas de Titans-MAC. B6 no ajusta parámetros, pero las dos entradas "
        "comprueban la protección porque eligen η con la validación y predicen ventanas de la "
        "campaña"
    ),
    "tests/training/test_rolling_retention.py::"
    "test_a_plateau_of_a_joint_fit_has_no_tables_to_release": (
        "recorre la retención v2 sobre la campaña reducida del fixture `base` para comprobar que "
        "la meseta no se regenera y su continuación sí, y la regeneración de predicciones se "
        "detiene con la protección"
    ),
    "tests/simulation/test_stage_report.py::"
    "test_component_contrasts_pair_each_variant_with_its_control_on_the_same_sessions": (
        "recorre la etapa de políticas reducida con GRPO, Dr. GRPO y GSPO y un aprendiz "
        "sustituido que no ajusta, porque el informe solo lee salidas confirmadas de la etapa y "
        "su ejecución se detiene con la protección"
    ),
    "tests/training/test_titans_walk_forward.py::"
    "test_window_builds_the_declared_predictor_changes_and_its_regeneration_repeats_them": (
        "ajusta una ventana Titans-MAC con PT1 y el optimizador que no modifica pesos para "
        "comprobar que los cambios del predictor entran en su identidad y que la regeneración los "
        "repite, y el ajuste se detiene con la protección"
    ),
    "tests/training/test_titans_campaign.py::"
    "test_an_extension_fit_builds_its_changes_and_its_carry_rebuilds_them": (
        "ejecuta el ajuste y el traslado de la campaña reducida con PT1 sobre la protección "
        "rebajada de su fixture, porque los dos ejecutores se detienen con ella"
    ),
    "tests/training/test_mars_titan_campaign.py::"
    "test_correction_frozen_parent_reproduces_the_carry_and_predicts_the_next_validation": (
        "traslada como padre congelado una ventana B6 de la campaña B reducida, que el fixture "
        "campaign_run ejecuta con la protección rebajada de permitted. carry_correction no ajusta, "
        "pero comprueba la protección porque predice ventanas de la campaña"
    ),
    "tests/training/test_campaign_a_joint.py::"
    "test_v2_joint_stop_pairs_each_case_and_the_selected_case_of_every_scope": (
        "ejecuta con run_campaign la ventana fold-006 de la v2 reducida con parada conjunta, que "
        "pasa la protección de la campaña y de cada trabajo con la protección rebajada de "
        "joint_campaign, aunque los ejecutores sustituidos no ajustan"
    ),
    "tests/training/test_campaign_a_joint.py::"
    "test_a_plateau_that_reports_another_precision_is_not_confirmed": (
        "ejecuta con run_campaign una meseta que informa de TF32 para comprobar que no se "
        "confirma, y pasa la protección con la rebajada de joint_campaign, aunque los ejecutores "
        "sustituidos no ajustan"
    ),
    "tests/training/test_modality_ablation_stage.py::"
    "test_masked_sources_keep_the_platform_of_the_base_they_compare_with": (
        "publica las fuentes de la ablación desde la salida que el fixture staged ejecuta con la "
        "protección rebajada y ejecutores que no ajustan, para comprobar la plataforma que "
        "guarda cada recibo y su coincidencia con la base (#543)"
    ),
    "tests/simulation/test_stage_report.py::"
    "test_the_report_records_the_platform_that_each_receipt_keeps": (
        "lee la salida de la etapa de políticas que el fixture stage_output ejecuta con "
        "learning_doubles y políticas de reglas fijas, para comprobar la plataforma de sus "
        "recibos y del informe (#543)"
    ),
    "tests/training/test_rolling_retention.py::"
    "test_aggregates_of_receipts_without_platform_need_the_declared_profile": (
        "recorre la campaña del fixture base, confirmada con la protección rebajada y pesos "
        "iniciales, con sus recibos sin plataforma para comprobar que los agregados solo se "
        "publican con el perfil declarado (#543)"
    ),
    "tests/test_suite_support.py::test_unguarded_fits_run_when_the_hold_allows_learning": (
        "la sesión de pytest hija lee una protección temporal que permite aprender para "
        "comprobar que la marca unguarded_fit deja de omitir, aunque la prueba sondeada no ajusta"
    ),
    "tests/training/test_campaign_launch_options.py::"
    "test_every_trainer_of_the_window_receives_the_declared_options_in_its_slot": _LAUNCH,
    "tests/training/test_campaign_launch_options.py::"
    "test_a_cut_campaign_resumes_every_job_with_the_same_options": _LAUNCH,
    "tests/training/test_campaign_launch_options.py::"
    "test_changed_options_reach_the_trainers_that_use_them": _LAUNCH,
    "tests/training/test_campaign_launch_options.py::"
    "test_an_undeclared_value_is_rejected_before_opening_views_or_output": (
        "comprueba que el motor rechaza el valor antes de abrir vistas o crear la salida, y el "
        "motor exige la protección antes de leer la campaña. Los mismos rechazos se comprueban "
        "sin la guarda con `check_campaign`"
    ),
    "tests/training/test_campaign_a_joint.py::"
    "test_neural_receipts_record_the_declared_kernels_in_their_identity": (
        "los recibos salen de la campaña reducida que el motor ejecuta con ejecutores "
        "sustituidos que no ajustan"
    ),
    "tests/training/test_reference_run_precision.py::"
    "test_resume_keeps_the_batch_precision_and_graph_step_it_started_with": (
        "la reanudación compara la identidad dentro de `run_reference_case`, después de su "
        "guarda, y el recorrido se detiene antes de leer lotes de ajuste"
    ),
    "tests/posttraining/test_masked_run.py::"
    "test_parent_and_adapter_run_with_the_strict_precision_of_the_parent_case": _ADAPTER,
    "tests/posttraining/test_masked_run.py::"
    "test_an_adapter_whose_process_allows_tf32_mid_run_fails_at_the_next_save": _ADAPTER,
    "tests/posttraining/test_masked_run.py::"
    "test_a_final_evaluation_under_tf32_is_not_confirmed": _ADAPTER,
    "tests/posttraining/test_publication_from_aggregates.py": (
        "completa la etapa de adaptadores con el optimizador que solo registra gradientes y "
        "recorre la retención v2, cuya regeneración de predicciones se detiene con la protección"
    ),
    "tests/training/test_rolling_retention.py::"
    "test_the_campaign_publication_reads_the_aggregates_with_portfolio_and_ablation": (
        "ejecuta cuatro ajustes CUDA de dos épocas sobre dos activos sintéticos, regenera "
        "sus predicciones y comprueba la publicación desde agregados con el historial real "
        "de esos ajustes técnicos"
    ),
    _WINDOWS + "test_resume_admission_preserves_confirmed_runtime": _WINDOW_UPDATES,
    _WINDOWS + "test_full_short_epochs_seal_configuration_before_first_update": _WINDOW_UPDATES,
    _WINDOWS
    + "test_six_published_routes_complete_fixture_updates_and_recover_checkpoint": _WINDOW_UPDATES,
    _WINDOWS + "test_resume_restores_exact_optimizer_rng_cursor_and_weights": _WINDOW_UPDATES,
    _WINDOWS
    + "test_selection_can_keep_initial_checkpoint_without_stopping_fixed_updates": _WINDOW_UPDATES,
    _WINDOWS + "test_timeout_is_partial_and_has_recoverable_confirmed_checkpoint": _WINDOW_UPDATES,
    _WINDOWS + "test_fp32_strict_is_applied_and_enforced_before_checkpoint": _WINDOW_UPDATES,
    _WINDOWS
    + "test_selected_session_artifacts_reuse_predictions_and_completed_resume_verifies_them": (
        _WINDOW_UPDATES
    ),
    _WINDOWS + "test_published_precomputed_recipe_completes_fixture_updates_and_checkpoint": (
        _WINDOW_UPDATES
    ),
}


def lowers_learning_hold(key):
    """Marca que omite la prueba o el módulo `key` del inventario mientras rige el bloqueo.

    La protección se lee al importar el módulo de pruebas, antes de que ningún fixture la
    sustituya, así que decide la protección real de la sesión.
    """
    if key not in INVENTORY:
        raise KeyError(f"{key} no figura en el inventario de pruebas que rebajan la protección")
    return pytest.mark.skipif(
        learning_blocked(),
        reason=f"Bloqueo de aprendizaje vigente: {INVENTORY[key]} (inventario de #546)",
    )


def allowed_hold(path):
    """Escribir en `path` una protección temporal permitida y devolver su ruta.

    Solo procede en recorridos cuyo aprendizaje está sustituido por dobles que no modifican
    pesos. El gancho de `torch.optim` que instala `tests/conftest.py` con la protección real
    sigue omitiendo cualquier paso de optimizador.
    """
    path = Path(path)
    path.write_text(json.dumps({"training_allowed": True}), encoding="utf-8")
    return path


# Pruebas que escriben una protección con un valor que solo se conoce al ejecutar, o el texto
# de una permitida en C++. Prueban el lector de la protección o el fixture `learning_hold`.
READERS = {
    "native/tests/learning_hold_tests.cpp": (
        "prueba el lector nativo con protecciones temporales permitidas, bloqueantes y ambiguas"
    ),
    "tests/conftest.py": (
        "el fixture learning_hold escribe la protección que pide cada prueba: False bloquea y "
        "True solo lo piden las pruebas de INVENTORY"
    ),
    "tests/evaluation/test_final_test_opening.py": (
        "la apertura del test final exige la protección levantada y la prueba la simula con un "
        "abridor que solo escribe un texto, sin aprender ni leer datos"
    ),
    "tests/posttraining/test_learning_hold_entrypoints.py": (
        "prueba que los puntos de entrada del postentrenamiento leen la protección"
    ),
    "tests/training/test_learning_hold.py": "prueba el lector de la protección en Python",
    "tests/training/test_learning_hold_guards.py": (
        "recorre cada punto de entrada con la protección permitida o ausente hasta el doble que "
        "sustituye su ajuste, que solo cuenta la llamada"
    ),
}

# Motivos de los archivos que piden una protección temporal permitida o redirigen la variable.
# Ninguno ajusta parámetros: el aprendizaje es un doble y el gancho de `torch.optim` que
# `tests/conftest.py` instala con la protección real sigue omitiendo cualquier paso.
_DEFINES = "define learning_doubles con allowed_hold"
_FIXTURES = (
    "learning_hold y learning_doubles apuntan la variable a su protección temporal, y el motivo "
    "de la omisión de `unguarded_fit` la nombra"
)
_CAMPAIGN = (
    "recorre una campaña o una etapa con ejecutores dobles que escriben predicciones o recibos "
    "sin ajustar ningún modelo"
)
_RECORDER = (
    "ajusta con el optimizador registrador, que calcula gradientes sin modificar pesos y no "
    "hereda de torch.optim.Optimizer"
)
_ESTIMATOR = (
    "sustituye el estimador, xgb.train o el solver de Ridge por un doble antes de ajustar, así "
    "que no estima ningún parámetro"
)
_SCRIPTED = "la etapa de políticas usa ScriptedLearner, sin redes ni optimizador"
_LAUNCHER = "el lanzador ejecuta un binario falso sin aprendizaje"
_POSTTRAINING = (
    "admite los puntos de entrada del postentrenamiento, cuyo AdamW sustituye el registrador"
)
_WARMUP = (
    "el entrenador PPO de Python avanza menos transiciones que el calentamiento y el "
    "recorrido, así que no actualiza, y el gancho de torch.optim sigue activo"
)
_REAL_HOLD = "lee la protección real por su ruta y se detiene si no bloquea o cambia su huella"
_RETIRED = "comprueba que la campaña B retirada se rechaza antes de llegar a ningún ejecutor"
_COUNTED = "las pasadas de ajuste solo cuentan actualizaciones y la validación es fija"
_CLOSES = "solo apunta la variable a una protección que bloquea"
_BINARY = (
    "pasa al binario la protección de cada prueba, que bloquea salvo en las pruebas de INVENTORY"
)
_READER = "prueba el lector de la protección con protecciones temporales y ausentes"
_MESSAGE = "solo nombra la variable en el motivo de la omisión mientras rige el bloqueo"
_OPENING = READERS["tests/evaluation/test_final_test_opening.py"]
_REUSES = (
    "reutiliza la protección permitida de un fixture de dobles para regenerar o trasladar "
    "predicciones de ventanas con los pesos iniciales"
)

# Número de referencias de cada archivo a los dobles (`learning_doubles`,
# `learning_doubles_module`, `allowed_hold`) y a la variable o la ruta de la protección. La
# guarda exige que coincidan, así que un uso nuevo necesita declararse aquí con su motivo.
DOUBLES = {
    "tests/tooling/test_variant_screening_study.py": (
        2,
        "el motor de la criba llama a ejecutores dobles que solo escriben su run.json, también "
        "el ajuste que espera la época común, sin construir modelos ni ejecutar pasos de "
        "optimizador",
    ),
    "tests/models/test_qlib_lightgbm.py": (
        1,
        "sustituye el módulo lightgbm por un doble que recorre las rondas y devuelve un booster "
        "de predicción fija, así que no estima ningún parámetro",
    ),
    "tests/models/test_session_ridge.py": (
        1,
        "sustituye `Ridge` de scikit-learn por un doble que predice una columna fija según su "
        "alpha, así que no estima ningún coeficiente",
    ),
    "tests/training/test_signal_ceiling.py": (
        1,
        "Ridge y LightGBM son dobles de predicción fija sobre la vista corta del fixture, así "
        "que el origen se recorre entero sin ajustar ningún modelo",
    ),
    "tests/training/test_signal_ceiling_cli.py": (
        1,
        "la verificación de las cohortes y cada origen son dobles que devuelven informes "
        "sintéticos, así que el script solo une sus series",
    ),
    "tests/training/test_common_control_windows.py": (
        4,
        "la regla no ajusta nada y el ajuste de LightGBM es un doble de predicción fija sobre la "
        "vista corta del fixture",
    ),
    "tests/training/test_salience_correction_window.py": (
        1,
        "recorre la corrección E3 con un padre de pesos literales, sin construir entrenadores "
        "ni ejecutar pasos de optimizador",
    ),
    "tests/training/test_regime_gate_correction_window.py": (
        1,
        "recorre la corrección E4 con un padre de pesos literales, sin construir entrenadores "
        "ni ejecutar pasos de optimizador",
    ),
    "tests/training/test_area_correction_bounded.py": (1, _RECORDER),
    "tests/conftest.py": (2, _DEFINES),
    "tests/training/test_window_variants.py": (
        1,
        "el fixture `allowed_child` escribe la protección permitida. Las pruebas que actualizan "
        "pesos están en el inventario y las demás se rechazan antes del primer forward",
    ),
    "tests/tooling/test_neural_plasticity_analysis.py": (
        1,
        "analiza predicciones y checkpoints mínimos escritos por el fixture, sin abrir "
        "datos reales ni ejecutar un optimizador",
    ),
    "tests/tooling/test_local_capture_study.py": (
        1,
        "analiza los controles de captura sobre archivos sintéticos de multiseed_fixture, "
        "sin construir entrenadores ni ejecutar pasos de optimizador",
    ),
    "tests/tooling/test_quantile_multiseed_analysis.py": (
        3,
        "comprueba el análisis, su entrada CLI y los rechazos sobre predicciones y checkpoints "
        "sintéticos de complete_fixture, sin entrenamientos ni fuentes reales",
    ),
    "tests/tooling/test_quantile_point_analysis.py": (
        1,
        "analiza objetivos puntuales con manifiestos y predicciones sintéticos de "
        "point_fixture, sin ajustar los modelos ni calcular predicciones reales",
    ),
    "tests/tooling/test_neural_plasticity_study.py": (
        1,
        "prepara un corpus técnico, conecta el entrenador real con un optimizador registrador "
        "y solicita pausa antes de cualquier actualización",
    ),
    "tests/tooling/test_refinement_study.py": (
        3,
        "comprueba sellos, reconstrucción congelada y un recorrido de fixtures con "
        "optimizadores registradores, sin abrir fuentes de una campaña real",
    ),
    "tests/tooling/test_combined_memory_numerics.py": (
        4,
        "comprueba rutas, publicación y rechazos con corpus e inferencia simulados, "
        "sin ajustes y con un gancho que impide cualquier paso de optimizador",
    ),
    "tests/tooling/test_combined_memory_queue.py": (
        5,
        "comprueba pausas, plazo y cierre de procesos con trabajos y ejecutores simulados",
    ),
    "tests/tooling/test_combined_memory_study.py": (
        2,
        "comprueba el catálogo y sus argumentos con ejecutores sustituidos antes del ajuste",
    ),
    "tests/learning_traces/test_native_policy_traces.py": (
        1,
        "la etapa de políticas se detiene al faltar la capacidad de trazas o las vistas, sin "
        "binarios ni fuentes abiertas",
    ),
    "tests/models/test_external_cache_cuda.py": (1, _ESTIMATOR),
    "tests/models/test_ridge_statistics_cuda.py": (1, _ESTIMATOR),
    "tests/posttraining/campaign_fixture.py": (1, _CAMPAIGN),
    "tests/posttraining/conftest.py": (1, _POSTTRAINING),
    "tests/posttraining/test_readout_adapters.py": (1, _RECORDER),
    "tests/simulation/rl_stage_fixture.py": (1, _CAMPAIGN),
    "tests/simulation/test_adaptive_campaign.py": (1, _LAUNCHER),
    "tests/simulation/test_campaign_stage.py": (24, _SCRIPTED),
    "tests/simulation/test_native_ppo_launcher.py": (1, _LAUNCHER),
    "tests/simulation/test_policy_recovery.py": (2, _WARMUP),
    "tests/simulation/test_stage_report.py": (5, _SCRIPTED),
    "tests/tooling/test_observatory_window_campaigns.py": (1, _SCRIPTED),
    "tests/training/candidate_memory_check.py": (1, _RECORDER),
    "tests/training/cuda_candidate_run_check.py": (1, _RECORDER),
    "tests/training/cuda_mars_titan_run_check.py": (1, _RECORDER),
    "tests/training/cuda_regeneration_check.py": (2, _RECORDER),
    "tests/training/test_campaign_a_joint.py": (5, _CAMPAIGN),
    "tests/training/test_campaign_chain.py": (1, _RETIRED),
    "tests/training/test_campaign_slots.py": (1, _CAMPAIGN),
    "tests/training/test_campaign_storage.py": (2, _CAMPAIGN),
    "tests/training/test_candidate_accumulation.py": (1, _RECORDER),
    "tests/training/test_candidate_run.py": (1, _RECORDER),
    "tests/training/test_candidate_walk_forward.py": (2, _RECORDER),
    "tests/training/test_cm_v1_factorial.py": (3, _RECORDER),
    "tests/training/test_external_shared_cuda.py": (1, _ESTIMATOR),
    "tests/training/test_joint_stop_trainers.py": (6, _COUNTED),
    "tests/training/test_mars_titan_correction.py": (4, _RECORDER),
    "tests/training/test_mars_titan_run.py": (10, _RECORDER),
    "tests/training/test_mars_session_inputs.py": (1, _RECORDER),
    "tests/training/test_mars_titan_walk_forward.py": (3, _RECORDER),
    "tests/training/test_masked_campaign.py": (1, _CAMPAIGN),
    "tests/training/test_masked_campaign_joint.py": (1, _CAMPAIGN),
    "tests/training/test_masked_reference_cuda.py": (1, _RECORDER),
    "tests/training/test_masked_reference_run.py": (2, _RECORDER),
    "tests/training/test_masked_tabular.py": (2, _ESTIMATOR),
    "tests/training/test_modality_ablation_carry.py": (1, _RECORDER),
    "tests/training/test_modality_ablation_native.py": (1, _RECORDER),
    "tests/training/test_modality_ablation_stage.py": (1, _CAMPAIGN),
    "tests/training/test_nvtx_parity.py": (1, _RECORDER),
    "tests/training/test_online_reference.py": (3, _RECORDER),
    "tests/training/test_quantile_head_control.py": (8, _CAMPAIGN),
    "tests/training/test_ridge_shared_cuda.py": (1, _ESTIMATOR),
    "tests/training/test_tabular_retention.py": (2, _ESTIMATOR),
    "tests/training/test_titans_campaign.py": (1, _RECORDER),
    "tests/training/test_titans_walk_forward.py": (5, _RECORDER),
}
REDIRECTIONS = {
    "tests/memory/test_lesion_inference.py": (1, _CLOSES),
    "tests/conftest.py": (3, _FIXTURES),
    "tests/tooling/test_cuda_test_runner.py": (
        2,
        "pasa una protección temporal cerrada al proceso de prueba y comprueba que la hereda",
    ),
    "tests/tooling/test_refinement_latency_study.py": (1, _CLOSES),
    "tests/tooling/test_refinement_scaling_analysis.py": (1, _CLOSES),
    "tests/tooling/test_refinement_study.py": (2, _CLOSES),
    "tests/tooling/test_combined_memory_numerics.py": (1, _CLOSES),
    "tests/tooling/test_combined_memory_queue.py": (1, _CLOSES),
    "tests/tooling/test_combined_memory_study.py": (1, _CLOSES),
    "tests/tooling/test_memory_microstudy.py": (1, _CLOSES),
    "tests/tooling/test_memory_microstudy_analysis.py": (1, _CLOSES),
    "tests/evaluation/test_final_test_opening.py": (1, _OPENING),
    "tests/native/test_optimizer_steps_hold.py": (1, _CLOSES),
    "tests/posttraining/campaign_fixture.py": (1, _CAMPAIGN),
    "tests/posttraining/test_learning_hold_entrypoints.py": (2, _READER),
    "tests/posttraining/test_publication_from_aggregates.py": (
        4,
        "reutiliza la protección permitida de la campaña base del fixture para completar la "
        "etapa de adaptadores con el optimizador registrador y recorrer la retención v2",
    ),
    "tests/posttraining/test_readout_adapters.py": (1, _RECORDER),
    "tests/simulation/rl_stage_fixture.py": (1, _CAMPAIGN),
    "tests/simulation/test_native_learning_hold.py": (1, _BINARY),
    "tests/simulation/test_native_policy_tapes.py": (4, _BINARY),
    "tests/simulation/test_native_ppo_launcher.py": (3, _CLOSES),
    "tests/simulation/test_native_ppo_runner.py": (1, _MESSAGE),
    "tests/simulation/test_policy_device_parity.py": (1, _CLOSES),
    "tests/test_learning_hold_lowering.py": (1, _CLOSES),
    "tests/training/test_campaign_a_joint.py": (1, _CAMPAIGN),
    "tests/training/test_candidate_walk_forward.py": (1, _RECORDER),
    "tests/training/test_financial_run.py": (1, _CLOSES),
    "tests/training/test_learning_hold.py": (4, _READER),
    "tests/training/test_mars_titan_run.py": (1, _RECORDER),
    "tests/training/test_modality_ablation_carry.py": (
        3,
        _REUSES + ". La prueba inventariada prepara además la corrección B6, que no ajusta",
    ),
    "tests/training/test_modality_ablation_native.py": (3, _REUSES),
    "tests/training/test_modality_ablation_stage.py": (9, _REUSES),
    "tests/training/test_prediction_regeneration.py": (4, _REUSES),
    "tests/training/test_rolling_retention.py": (9, _REUSES),
    "tests/training/test_titans_campaign.py": (1, _RECORDER),
    "tests/training/test_titans_walk_forward.py": (1, _RECORDER),
    "tests/training/test_window_variant_cli.py": (1, _CLOSES),
    "tests/training/test_window_variants.py": (
        2,
        "apunta la variable a la protección permitida de `allowed_child` y, en la prueba del "
        "orden de comprobaciones, a una que bloquea",
    ),
    "benchmarks/hierarchical_bridge_cuda.py": (1, _REAL_HOLD),
    "benchmarks/nonlinear_bridge_cuda.py": (
        2,
        "lee la protección de MARS_TITAN_TRAINING_HOLD o la real, registra su huella antes y"
        " después y la pasa sin cambios a cada proceso hijo, que no calculan gradientes ni"
        " usan optimizador",
    ),
    "benchmarks/regional_decision_cuda.py": (1, _REAL_HOLD),
    "benchmarks/regional_program_cuda.py": (1, _REAL_HOLD),
    "benchmarks/regional_program_session_cuda.py": (1, _REAL_HOLD),
    "reports/engineering/regional-geometry-20261022/cost.py": (1, _REAL_HOLD),
}
