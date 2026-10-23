# Arquitectura propuesta

MARS-TITAN se diseña como una comparación reproducible de componentes. La estructura separa las responsabilidades necesarias para implementar y comparar cada componente científico.

La [dirección arquitectónica vigente](../research/titans-mac-architecture.md) separa la referencia GRU con banco episódico, Transformer compacto, Titans-MAC y MARS-TITAN con ampliaciones. La [especificación episódica](../research/candidate-architecture.md) conserva retención, recurrencia y escritura del candidato anterior. Las comparaciones usan todo el universo admisible de su edición con lectura por lotes, sin un límite de 128 activos.

La [revisión de integración](../research/system-integration.md) sitúa las ampliaciones en una vista común de información y un ciclo cronológico de decisión y maduración. Los componentes de lectura, actualización asociativa, escritura episódica y replay conservan responsabilidades distintas. El componente GRU de #296, las referencias existentes y los módulos aislados no acreditan la integración de Titans-MAC ni una comparación científica de las nuevas variantes.

## Límites entre componentes

| Componente | Entrada | Salida y responsabilidad |
| --- | --- | --- |
| Datos | Fuentes y reglas de disponibilidad. | Muestras y máscaras auditables. No conoce modelos ni calcula resultados financieros. |
| Objetivos | Precios futuros usados solo como etiquetas y coeficientes temporales. | Retorno bruto/residual, intervalo y fecha de maduración. |
| Codificadores | Ventanas históricas y modalidades disponibles. | Vectores compactos. Sin acceso al objetivo futuro. |
| Memoria | Representaciones observadas, estado anterior y eventos autorizados de actualización. | Estado consultable y registro de lectura/escritura. Sin acceso libre a tablas de etiquetas. |
| Predicción | Representación actual y lectura de memoria. | Estimación puntual y cuantiles o escala. No ejecuta operaciones. |
| Calibración | Predicciones y etiquetas de un tramo reservado. | Intervalos y política de abstención fijados antes de evaluar. |
| Entrenamiento | Configuración, muestras permitidas y particiones. | Parámetros, estado inicial, curvas de ajuste y trazabilidad. |
| Evaluación | Predicciones inmutables, etiquetas maduras y política experimental. | Métricas, simulación y tablas. No reajusta el modelo usando el test. |

## Modelo compacto de partida

Las referencias recurrentes, DLinear y el Transformer compacto comparten en `MultimodalReference` las proyecciones de modalidades, la fusión y la cabeza, así que el contraste entre familias solo cambia el codificador de precios. Titans-MAC añade atención cercana, memoria neuronal actualizable y parámetros persistentes aprendidos. El texto conserva embeddings congelados con procedencia y fecha documentadas. Las ausencias de la edición histórica mantienen sus máscaras, sin cambiar las filas por arquitectura.

La primera memoria del candidato será global. Una consulta dependerá de la representación actual y recuperará un vector de contexto que la cabeza combine con el estado temporal. El estado y sus operaciones `read`, `update`, `reset` y `snapshot` deben tener responsabilidades separadas. Estas son interfaces previstas para el candidato. La memoria nativa ya implementada pertenece a los comparadores financieros y tiene otro ámbito y dimensiones. Después se podrá comparar la separación mercado/sector/activo manteniendo control de capacidad total.

La memoria neuronal aplica la pérdida asociativa, gradientes, momentum y olvido dependientes de la entrada del núcleo Titans-MAC. El banco episódico sigue siendo una ampliación separada. Los pesos rápidos, los parámetros persistentes de Titans, los episodios y el estado de trabajo tienen ciclos y checkpoints distintos. Se documentará qué gradientes se propagan y qué estado se desacopla al persistirlo. Una regla delta aislada no sustituye esa correspondencia con el artículo.

La [lectura de bancos neuronales](../experiments/bank-readout-v1.md) combina las salidas de tres redes históricas por flujo. Su [variante de escritura](../experiments/bank-write-v1.md) contrasta parciales por banco y su media compartida, con tres momentum por capa y recuperación identificada. La traslación común anterior sigue como referencia y valor por defecto. Estas opciones no sustituyen los episodios ni convierten el grafo regional en memoria histórica. Sus comprobaciones técnicas y costes no acreditan una mejora predictiva.

La [jerarquía del puente episódico](../experiments/hierarchical-evidence-bridge-v1.md) se ejecuta después de MAC. Cuatro vistas, dos grupos y una raíz comunican representaciones del mismo snapshot. Sus auxiliares se reinician por lectura y no añaden bancos ni pesos rápidos. El árbol, la estrella y el control sin realimentación reutilizan los parámetros regionales. Esta opción fija y desactivada tiene [comprobaciones y costes propios](../../reports/engineering/hierarchical-bridge-20261022/README.md), separados de la fusión anterior a MAC y de una selección aprendida.

### Codificación y fusión de eventos

La fusión de las cinco entradas se define una sola vez, en `fuse_modalities` de [`multimodal.py`](../../src/mars_titan/models/baselines/multimodal.py). La usan `MultimodalReference` y `FinancialPredictor`, que calcula el token fusionado antes de Titans-MAC. Con la política estricta concatena la representación de precios y las cuatro proyecciones. Con `historical_masked_2000_v1` anula después del sesgo la proyección de cada modalidad ausente y añade los cinco bits de presencia a la primera capa (`zero_after_projection_then_concat_presence`). Los precios no se multiplican porque el lector los exige y sus huecos de mercado ya llegan anulados por `gate_price_window`.

Las cuatro variantes del adaptador (`transformer_direct`, `mac_disabled`, `mac_frozen` y `mac_online`) construyen codificador, proyecciones, fusión y cabeza con la misma semilla antes de crear la memoria. Con el mismo lote reciben las mismas muestras y producen el mismo token bit a bit, y solo cambia lo que ocurre después de él. La GRU y el Transformer reinician su estado en cada ventana, de modo que un cambio de pesos no obliga a reconstruir ningún estado recurrente. El único estado que pasa de una decisión a la siguiente es el de Titans-MAC, cuyos gradientes y serialización se describen en la [dirección arquitectónica](../research/titans-mac-architecture.md#estados-y-disponibilidad).

| Criterio de MT-017 | Evidencia |
| --- | --- |
| Contrato estable de salida | `MultimodalReference.configuration` y `FinancialConfig.identity` registran fusión, cabeza y codificador. Un contrato MAC distinto no se carga con la misma semilla y formas (`test_same_seed_and_shapes_do_not_allow_loading_another_mac_contract` en [`test_financial_adapter.py`](../../tests/models/titans/test_financial_adapter.py)) y la fusión estricta conserva su construcción y su contrato (`test_disabled_option_keeps_the_strict_construction_and_contract` en [`test_masked_reference_fusion.py`](../../tests/models/test_masked_reference_fusion.py)). |
| Causalidad de las ventanas | En los lectores, `test_future_rows_and_absences_do_not_change_a_window` ([precios](../../tests/data/test_price_windows.py)), `test_future_news_and_revisions_do_not_change_past_sample` ([muestras](../../tests/data/test_samples.py)), `test_chart_only_depends_on_past_window_and_is_reproducible` ([gráficos](../../tests/data/test_charts_cutoff.py)) y `test_macro_memory_limit_and_future_input_fail_closed` ([macro](../../tests/data/test_cohort_contexts.py)). En los modelos, `test_future_suffix_cannot_change_prefix_or_receive_its_gradient` ([Transformer](../../tests/models/test_compact_transformer.py)) y `test_future_segments_do_not_change_previous_outputs_or_input_state` ([MAC](../../tests/models/titans/test_mac.py)). |
| Ausencia explícita y comprobable | Gradiente nulo y relleno irrelevante en las cinco familias de referencia ([`test_masked_reference_fusion.py`](../../tests/models/test_masked_reference_fusion.py)). En el adaptador, `test_absent_modalities_are_zeroed_after_bias_and_presence_enters_common_fusion` y `test_each_present_modality_reaches_its_projection_and_an_absent_one_gets_no_gradient` ([`test_common_fusion.py`](../../tests/models/titans/test_common_fusion.py)). |
| Variante sin memoria con las mismas entradas y componentes | `test_the_four_controls_share_the_common_modules_and_emit_the_same_token` y `test_reference_and_predictor_name_the_same_fusion` ([`test_common_fusion.py`](../../tests/models/titans/test_common_fusion.py)), además de `test_reference_reproduces_the_financial_predictor_fusion_exactly`. |
| Admisión de texto, gráficos y fundamentales | [Máscaras históricas](../data/historical-input-masks.md), [entradas del corpus](../data/historical-corpus-inputs.md) y [auditoría de los codificadores](../data/pretraining-audit.md). |
| Codificador común de M0–M3 y de CM-v1 con sus factores apagados | M0 conserva el refinador con lectura vacía y nunca escribe (`test_m0_keeps_the_refiner_with_an_empty_read_and_never_writes` en [`test_financial_session_gru.py`](../../tests/memory/test_financial_session_gru.py)). La paridad de CM-v1 se sigue en [#293](https://github.com/GonxKZ/mars-titan/issues/293). |
| Coste en `cuda:0` | La refactorización no cambia el cálculo: la traza FP32 guardada del adaptador, con salidas, gradientes y estado, se reproduce bit a bit (`test_disabled_component_keeps_fp32_outputs_gradients_and_state_bit_for_bit` en [`test_memory_stability.py`](../../tests/models/titans/test_memory_stability.py)). Caudal y memoria de las cuatro variantes con la fusión con presencia, en FP32 estricto sobre US y CN, en [`vram-throughput.json`](../../reports/engineering/campaign-kernels-20261009/vram-throughput.json) (por ejemplo, `transformer_direct` con 9.225 muestras/s y 890 MiB reservados en una RTX 4070 Laptop). |
| Rasgos base, R fija y vista versionados | `test_codec_rejects_another_input_contract` y `test_encoding_verification_rejects_changed_contract` ([códec](../../tests/models/candidate/test_episode_codec.py)), y `test_codec_from_another_view_is_rejected_before_creating_a_session` ([sesión](../../tests/memory/test_financial_session.py)). |

La guía de la tarea preveía `models/events.py`, `models/fusion.py` y `configs/models/event-encoder.yaml`. No se han creado porque los codificadores, la fusión y su configuración ya viven en las referencias y en el adaptador. Moverlos cambiaría los nombres de los parámetros guardados y las identidades que comparan checkpoints. La selección de documentos antes de agregar (I10) sigue sin conectar y se registra en [#366](https://github.com/GonxKZ/mars-titan/issues/366).

## Sorpresa y régimen

La puntuación propuesta combina un error predictivo **maduro**, una anomalía calculada sobre historia y un indicador de relevancia económica cuya fuente sea verificable. Se normaliza cada componente con estadísticas del pasado y se estudia cada uno por separado. Si no hay expectativas de consenso observadas en tiempo real, no se llamará «sorpresa de beneficios» a la diferencia frente a un valor estimado retrospectivamente.

No se fijan pesos supuestamente óptimos sin experimento. El umbral y los pesos se eligen con validación. El registro incluirá escrituras por periodo, magnitud de actualización y saturación. La escritura aleatoria o uniforme con igual presupuesto sirve para comprobar si el efecto proviene de seleccionar eventos o solo de reducir ruido y cómputo.

El régimen inicial puede usar volatilidad y tendencia retrospectivas, con umbrales estimados en entrenamiento. Si se añade un modelo de estados ocultos, se utilizarán probabilidades **filtradas**, no estados suavizados con observaciones futuras. La identificación de un régimen sirve para condicionar o analizar predicciones, sin convertirlo en una explicación causal del mercado.

## Incertidumbre y abstención

La primera salida incierta puede ser una cabeza de cuantiles. Se medirán pérdida de cuantiles, cobertura y anchura. NLL solo será aplicable si otra variante define una densidad predictiva. El núcleo calibra en un tramo distinto del ajuste y del test y después congela el calibrador. Una extensión ACI podrá adaptarse de forma secuencial con etiquetas maduras bajo una política fijada antes de evaluar, sin selección de hiperparámetros durante el test. Ensambles pequeños o MC dropout son alternativas de coste a medir. Ninguna produce por sí sola una garantía de cobertura temporal. La decisión de abstenerse se evalúa a cobertura comparable y con exposición económica explícita.

## Python, C y C++/CUDA

Python mantiene las referencias científicas y parte de la preparación. `native/` contiene C++20 para contabilidad, simulación por lotes, filtrado HMM, memoria episódica, replay, políticas y recuperación, además de los ejecutables `mars-titan-sim` y `mars-titan-ppo`. Estos componentes no constituyen una implementación del candidato. La revisión de integración distingue qué contratos reutilizar y qué semánticas financieras deben permanecer separadas.

La ruta intensiva futura del candidato se propone como un ejecutable C++20 con LibTorch y bibliotecas matemáticas existentes, con una referencia de corrección y artefactos comparables. Un kernel propio solo se justifica cuando un perfil identifique un coste que no resuelvan mejor esas primitivas. Se compararán resultados, gradientes, tipos, formas, recuperación y errores de dispositivo antes de sustituir una referencia. El uso de C++ no acredita por sí solo una aceleración de operaciones que ya ejecutan código nativo.

El diseño evita servicios distribuidos, colas remotas y un despliegue en producción. Los artefactos de experimento son archivos locales con un registro estructurado. Se incorporará un gestor adicional solo si resuelve una necesidad observada.
