# Órdenes de línea de comandos

Los guiones son puntos de entrada de línea de comandos y la lógica reutilizable vive en `src/mars_titan/`. Se ejecutan con `uv run --locked python scripts/<guion>.py --help`. Las órdenes que ajustan modelos respetan la protección del aprendizaje descrita en la [guía de pruebas](../tests/README.md#protección-del-aprendizaje).

| Guion | Qué hace | Documento |
| --- | --- | --- |
| **Repositorio y bibliografía** | | |
| [`check_repository.py`](check_repository.py) | Enlaces, metadatos, bibliografía, órdenes documentadas y tabla de variables de entorno | [Reproducibilidad](../docs/engineering/reproducibility.md) |
| [`fetch_references.py`](fetch_references.py) | Descarga el catálogo bibliográfico y deja un manifiesto por referencia. Usa `reference_library.py` | [Referencias](../docs/references/README.md) |
| [`refresh_public_sources.py`](refresh_public_sources.py) | Instantáneas nuevas de fuentes públicas en `data/external/`. Valida formatos con `public_source_formats.py` | [Actualización de fuentes públicas](../docs/data/public-source-updates.md) |
| [`list_slow_tests.py`](list_slow_tests.py) | Regenera `tests/slow_tests.txt` a partir de informes JUnit | [Guía de pruebas](../tests/README.md#marcas-de-selección) |
| **Datos y campañas** | | |
| [`prepare_corpus_targets.py`](prepare_corpus_targets.py) | Objetivos residuales con un solo proceso por destino | [Esta guía](#objetivos-residuales-con-un-solo-proceso) |
| [`run_masked_campaign.py`](run_masked_campaign.py) | Campaña con máscaras desde 2000: comprobar, preparar, ejecutar, publicar y medir | [Esta guía](#campaña-con-máscaras-desde-2000) |
| [`run_titans_walk_forward.py`](run_titans_walk_forward.py) | Una ventana walk-forward de Titans-MAC con predicciones por fila | [Entrenador cronológico de Titans](../docs/engineering/titans-chronological-trainer.md) |
| [`run_quantile_head_control.py`](run_quantile_head_control.py) | Control de la cabeza de cuantiles | [Cabeza de cuantiles](../docs/engineering/quantile-head.md) |
| [`simulate_early_stopping.py`](simulate_early_stopping.py) | Reglas de parada sobre historiales de validación ya registrados | [Esta guía](#simulación-de-la-parada-temprana) |
| [`train_budget_probe.py`](train_budget_probe.py) | Medición supervisada breve de recursos, sin tocar la prueba final | [Presupuesto de la campaña](../reports/resources/campaign-budget.md) |
| **Simulación y refuerzo** | | |
| [`run_native_ppo.py`](run_native_ppo.py) | PPO o KLPO nativo con una carga GPU o un diagnóstico CPU explícito | [PPO nativo](../docs/engineering/native-ppo.md) |
| [`run_financial_comparators.py`](run_financial_comparators.py) | Comparadores financieros con datos confirmados y prueba final cerrada | [Simulación persistente](../docs/engineering/persistent-simulation.md) |
| [`prepare_financial_scenarios.py`](prepare_financial_scenarios.py) | Escenarios sintéticos con una referencia analítica o un padre verificado | [PPO nativo](../docs/engineering/native-ppo.md) |
| [`prepare_named_benchmarks.py`](prepare_named_benchmarks.py) | Describe o prepara los benchmarks identificados | [Benchmarks identificados](../docs/engineering/named-benchmarks.md) |
| [`generate_episode_worlds.py`](generate_episode_worlds.py) | Escenarios separados con semillas de evaluación independientes | [Episodios experimentales](../docs/engineering/experimental-episodes.md) |
| [`explain_rl_decisions.py`](explain_rl_decisions.py) | Describe decisiones confirmadas a partir de sus trazas | [Adaptación en escenarios sintéticos](../docs/engineering/adaptive-rl.md) |
| **Evaluación** | | |
| [`export_campaign_comparison.py`](export_campaign_comparison.py) | Tablas agregadas y figuras de las campañas comparadas | [Análisis de campañas congeladas](../docs/engineering/campaign-comparison.md) |
| **Observatorio** | | |
| [`collect_observatory.py`](collect_observatory.py) | Recoge informes pequeños y publica los cambios | [Observatorio](../docs/engineering/observatory.md) |
| [`export_observatory.py`](export_observatory.py) | Instantánea pública del observatorio desde estados locales | [Observatorio](../docs/engineering/observatory.md) |
| [`export_learning_traces.py`](export_learning_traces.py) | Paquete del observatorio con las trazas del registrador de aprendizaje | [Observatorio](../docs/engineering/observatory.md) |
| [`serve_observatory.py`](serve_observatory.py) | Sirve el observatorio en local con cambios en directo | [Web](../site/README.md) |
| **Medidas de rendimiento** | | |
| [`benchmark_anchored_medoids.py`](benchmark_anchored_medoids.py) | Selector de medoids con NumPy o SciPy | [Retención anclada](../docs/experiments/mars_titan_cm_v1/anchored_retention.md) |
| [`benchmark_campaign_comparison.py`](benchmark_campaign_comparison.py) | Análisis completo de campañas con y sin reutilizar la política inicial | [Análisis de campañas congeladas](../docs/engineering/campaign-comparison.md) |
| [`benchmark_campaign_observatory.py`](benchmark_campaign_observatory.py) | Lectura incremental y exportación del observatorio con fuentes reales | [Observatorio de campañas](../docs/engineering/campaign-observatory.md) |
| [`benchmark_cm_mechanisms.py`](benchmark_cm_mechanisms.py) | Mecanismos C y M de CM-v1 con datos sintéticos en CPU | [Resultados de CM-v1](../docs/experiments/mars_titan_cm_v1/results.md) |
| [`benchmark_evaluation_scale.py`](benchmark_evaluation_scale.py) | Tiempo y memoria de la comparación walk-forward y de la cartera a escala real | [Informe de escala](../reports/engineering/evaluation-scale-20261009/README.md) |
| [`benchmark_native_batches.py`](benchmark_native_batches.py) | Recorrido contable nativo con 1, 2, 4 y 8 trabajadores | [Rendimiento por lotes](../reports/resources/batched-rl-performance.md) |
| [`benchmark_native_ppo.py`](benchmark_native_ppo.py) | Procesos PPO de diagnóstico con fuentes sintéticas y sin CUDA | [PPO nativo](../docs/engineering/native-ppo.md) |
| [`benchmark_observatory.py`](benchmark_observatory.py) | Exportación sintética del observatorio en CPU | [Observatorio](../docs/engineering/observatory.md) |
| [`summarize_native_quality.py`](summarize_native_quality.py) | Cobertura LCOV y complejidad de las funciones nativas propias | [Verificación nativa](../reports/resources/native-financial-verification.md) |
| **Ejemplos sintéticos** | | |
| [`example_document_index.py`](example_document_index.py) | Preparación documental de control, sin evidencia histórica | [Índice documental](../docs/engineering/document-index.md) |
| [`example_information_views.py`](example_information_views.py) | Vistas y consumidores sobre `CorpusDataset`, sin entrenar | [Vistas de información](../docs/engineering/information-views.md) |
| **Ediciones anteriores ([módulos históricos](../src/mars_titan/README.md#módulos-históricos))** | | |
| [`finish_real_campaign.py`](finish_real_campaign.py) | Cierre del análisis de la edición real ampliada | [Edición real ampliada](../docs/engineering/real-expanded-comparison.md) |
| [`run_adaptive_campaign.py`](run_adaptive_campaign.py) | Piloto, comparación y auditoría de la campaña RL sintética | [Adaptación en escenarios sintéticos](../docs/engineering/adaptive-rl.md) |
| [`prepare_adaptation_scenarios.py`](prepare_adaptation_scenarios.py) | Mundos técnicos de esa campaña con su reserva de auditoría | [Adaptación en escenarios sintéticos](../docs/engineering/adaptive-rl.md) |
| [`benchmark_adaptive_rl.py`](benchmark_adaptive_rl.py) | Medida por pares del RL nativo de esa campaña | [Adaptación en escenarios sintéticos](../docs/engineering/adaptive-rl.md) |

`public_source_formats.py` y `reference_library.py` son módulos auxiliares de `refresh_public_sources.py` y `fetch_references.py`, no órdenes.

## Mantenimiento del repositorio

Estas utilidades conservan la biblioteca, revisan el repositorio y capturan fuentes públicas. No implementan preparación de muestras, entrenamiento, modelos ni evaluación financiera.

```bash
uv run python scripts/fetch_references.py
uv run python scripts/check_repository.py
uv run python scripts/refresh_public_sources.py --list
uv run python scripts/refresh_public_sources.py --source fed_monetary_rss
```

El descargador bibliográfico usa los catálogos de referencias, conserva archivos previos y genera un registro local. El verificador comprueba enlaces locales, metadatos, correspondencia BibTeX, planificación y archivos excluidos.

El actualizador público requiere `curl` y escribe nuevas instantáneas en `data/external/`, fuera de Git. Su selección por defecto incluye ocho fuentes renovables. Usa límites de tiempo, tamaño y disco, valida el contenido y suspende un proveedor tras HTTP 403 o 429. Reutiliza los contenidos idénticos ya confirmados, envía peticiones condicionales cuando hay validadores y regenera `data/manifests/public-snapshots/index.json`. Los PDF fijos requieren selección explícita y `pdfinfo`. La [guía de actualización](../docs/data/public-source-updates.md) explica sus límites y la separación respecto a los datos experimentales.

## Objetivos residuales con un solo proceso

`prepare_corpus_targets.py` envuelve `mars_titan.training.corpus_targets.prepare_corpus_targets`. Sin `--execute` solo comprueba. Lee el manifiesto materializado, valida su política de entradas, cuenta activos por mercado, recibos ya confirmados y el estado del cerrojo, y no escribe ningún archivo ni calcula etiquetas.

```bash
uv run --no-sync python scripts/prepare_corpus_targets.py \
  --manifest <manifiesto materializado> --prepared <preparados> --output <supervisión> \
  --input-policy historical_masked_2000_v1 --target-factors <revisión de factores>
```

Con `--execute` toma un cerrojo exclusivo en `.<destino>.lock`, junto al destino y fuera de él, y lo mantiene durante toda la preparación. Un segundo proceso sobre el mismo destino termina con código 3 sin tocar nada. La reanudación por recibos de la función no cambia. Esta utilidad no se ha ejecutado sobre la edición real. Generar los objetivos reales corresponde a la etapa 2 de la [campaña](../docs/research/training-campaign-2000.md) y sigue sujeto a la verificación de la edición.

## Campaña con máscaras desde 2000

`run_masked_campaign.py` envuelve `mars_titan.training.masked_campaign`, `mars_titan.training.campaign_throughput` y `mars_titan.training.campaign_extensions`. `check` cuenta los trabajos de una variante sin leer datos, `prepare` crea las vistas por ámbito, `run` ejecuta y reanuda los trabajos con recibos, `sources` publica el manifiesto de fuentes de un ámbito, `throughput` mide el caudal de cada familia sin pasos de optimizador y `extensions` comprueba los recuentos de la declaración preparada de la GRU candidata, MARS-TITAN y CM-v1 sin activarla. `run` respeta el bloqueo de aprendizaje antes de empezar y antes de cada trabajo. La [campaña](../docs/research/training-campaign-2000.md#ejecución-y-recuperación) describe sus órdenes. Ninguna orden se ha ejecutado sobre la edición real.

```bash
uv run --no-sync python scripts/run_masked_campaign.py check \
  --campaign configs/baselines/historical-masked-campaign-a.json
```

## Control de la cabeza de cuantiles

`run_quantile_head_control.py` envuelve `mars_titan.training.quantile_head_control`. `check` cuenta los trabajos del [control declarado](../configs/baselines/quantile-head-control-us.json) sin leer datos, `run` ajusta o reanuda cada par del Transformer compacto con recibos y respeta el bloqueo de aprendizaje, y `decide` aplica la regla de retroceso de [#22](https://github.com/GonxKZ/mars-titan/issues/22) con las predicciones de validación confirmadas. La [guía de la cabeza](../docs/engineering/quantile-head.md#ejecutor-del-control) describe ventanas, contraste y coste estimado. Ninguna orden de ajuste se ha ejecutado.

```bash
uv run --no-sync python scripts/run_quantile_head_control.py check \
  --plan configs/baselines/quantile-head-control-us.json
```

## Simulación de la parada temprana

`simulate_early_stopping.py` aplica la selección de `mars_titan.training.selection` a los historiales de validación de ejecuciones ya confirmadas, sin modificarlas, con varias paciencias, mejoras mínimas y épocas mínimas, en modo individual y conjunto. Agrupa los ajustes por ventana, ámbito, etapa, semilla y caso, cuenta como censurado un ajuste que necesitaría épocas no registradas y escribe en la salida estándar épocas recorridas, pérdida relativa del estado elegido y huella de las fuentes. No entrena ni lee datos. Su resultado sobre las referencias de la edición ampliada anterior está en el [recibo de la simulación](../reports/engineering/early-stopping-history-20261009.json) y se discute en la [parada temprana opcional](../docs/research/walk-forward-2000.md#parada-temprana-opcional).

```bash
uv run --no-sync python scripts/simulate_early_stopping.py \
  data/interim/real-expanded-references-20261006 > early-stopping.json
```

Las pruebas de `tests/tooling/` protegen estos comportamientos de mantenimiento sin realizar peticiones de red.
