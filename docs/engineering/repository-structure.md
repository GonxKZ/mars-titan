# Auditoría de la estructura del repositorio

Auditoría del 10 de octubre de 2026 sobre `develop` en `f2eef70c`, para la [#440](https://github.com/GonxKZ/mars-titan/issues/440). Describe la estructura actual, los problemas medidos y una estructura objetivo con su plan de migración. No mueve ningún archivo. Con nueve PR abiertas que tocan casi todos los paquetes, cualquier movimiento ahora provocaría conflictos en todas ellas.

## Cómo se ha medido

- Líneas y archivos por paquete con un recorrido de `src/mars_titan/`.
- Dependencias entre paquetes contando las importaciones absolutas `mars_titan.<paquete>` de cada módulo. Las importaciones relativas se cuentan dentro de su paquete.
- Consumidores de cada módulo buscando su ruta con puntos y sus formas de importación en todos los archivos versionados (código, pruebas, scripts, benchmarks, configuraciones y documentación).
- Funciones con el mismo nombre en varios módulos y comparación del cuerpo de las que calculan huellas.

Los recuentos son de importaciones escritas, no de llamadas en ejecución.

## Estado actual

| Paquete | Archivos | Líneas | Responsabilidad declarada |
| --- | ---: | ---: | --- |
| `training` | 51 | 22.894 | Entrenadores, vistas walk-forward, objetivos del corpus, campaña, caudal y bloqueo de aprendizaje |
| `data` | 75 | 21.793 | Fuentes, disponibilidad temporal, muestras multimodales y objetivos |
| `simulation` | 23 | 8.298 | Contabilidad, reglas de mercado, cintas, políticas y etapa RL |
| `evaluation` | 17 | 6.947 | Métricas por sesión y comparación walk-forward |
| `memory` | 17 | 6.014 | Banco episódico, escrituras, memoria asociativa y sesiones financieras |
| `posttraining` | 15 | 5.080 | Padres congelados y matriz de adaptadores |
| `models/titans` | 13 | 3.530 | Núcleo Titans-MAC y lector episódico |
| `models/baselines` | 13 | 2.942 | Referencias |
| `environments` | 6 | 1.497 | Fuentes causales de cohortes y entorno predictivo |
| `observatory` | 4 | 1.460 | Recolección y publicación del observatorio |
| raíz del paquete | 5 | 1.253 | Sondas de recursos y perfilado |
| `episodes` | 8 | 1.196 | Episodios cronológicos y mundos sintéticos |
| `cm` | 5 | 910 | Mecanismos C y M de CM-v1 |
| `models` (raíz y `candidate`) | 9 | 1.157 | Cabeza de cuantiles, objetivos KLPO y adaptador de la GRU candidata |
| `calibration` | 2 | 216 | Calibración común de cuantiles |

Fuera de `src/`, los archivos versionados se reparten así: `reports` 490, `tests` 390, `docs` 221, `native` 165, `configs` 75, `scripts` 33, `site` 12, `benchmarks` 12, `data` 9, `thesis` 4. `experiments/` y `notebooks/` tienen un archivo cada una.

## Hallazgos

### 1. `training` funciona como centro del que dependen las capas inferiores

Hay nueve pares de paquetes con dependencia mutua y siete pasan por `training`:

| Par | Importaciones en cada sentido |
| --- | --- |
| `posttraining` ↔ `training` | 43 / 10 |
| `data` ↔ `training` | 8 / 137 |
| `memory` ↔ `training` | 5 / 18 |
| `simulation` ↔ `training` | 14 / 5 |
| `evaluation` ↔ `training` | 5 / 21 |
| `models/baselines` ↔ `training` | 4 / 18 |
| `models/titans` ↔ `training` | 1 / 22 |
| `environments` ↔ `training` | 4 / 9 |

Lo que las capas inferiores toman de `training` no es entrenamiento, sino contratos y guardas transversales:
- `cohort_contract`, `temporal_contract`, `partition_contract` y `prefix_eligibility`, que definen qué filas y entradas son válidas en cada instante;
- `corpus_inputs` y `corpus_targets`, que construyen entradas y objetivos;
- `learning_hold`, la guarda del bloqueo de aprendizaje, importada por `models` y `simulation`;
- `run_receipts`, `checkpoints` y `selection`.

Esto contradice la inversión de dependencias que pide el proyecto. Un modelo o un lector de datos no debería depender del paquete que orquesta su entrenamiento. En la práctica obliga a importar `training` para usar un modelo y hace que un cambio en un contrato temporal parezca un cambio de entrenamiento.

### 2. `training` mezcla cinco responsabilidades

Sus 51 módulos se agrupan en:
- **Contratos y guardas** (hallazgo 1).
- **Construcción del corpus:** `corpus_*`, `temporal_corpus`, `joint_temporal_corpus`, `tabular_corpus`, `external_corpus`, `information_inputs`, `predictive_inputs` y `target_factors`.
- **Ejecuciones por familia:** `reference_run`, `candidate_run`, `financial_run`, `mars_titan_run`, `predictive_run` y los tres `*_walk_forward`.
- **Búsqueda y selección:** `reference_search`, `tabular_search`, `temporal_search`, `reference_design` y `selection`.
- **Orquestación de campañas:**
  - `campaign_*`, `masked_campaign`, `real_campaign` y `reference_campaign`;
  - `storage_*`, `gpu_supervisor`, `experiment_resources` y las colas;
  - `modality_ablation_stage` y `cm_v1_factorial`.

El nombre del paquete no deja ver ninguno de esos dominios.

### 3. `data` es plano con 75 módulos

Las familias se distinguen solo por prefijo: `news_*` (8), `macro_*` (18), `china_*` (5), `unadjusted_*` (3), `cohort_*` (6), `corpus_*` (3) y codificación (`embeddings`, `embedding_placement` y `charts`). Los prefijos ya describen subpaquetes naturales.

### 4. `simulation` junta el simulador de mercado con la etapa RL

`market`, `market_rules`, `portfolio`, `environment`, `session_prices`, `reconstructed_tape`, `window_tapes` y `replay` forman el simulador contable. `algorithms`, `training`, `policy_plan`, `adaptive_campaign`, `campaign_stage`, `campaign_receipts` y los `native_*` forman la etapa RL. Además, `environments` (entorno predictivo) y `episodes` (episodios y mundos sintéticos) tratan conceptos vecinos con otros nombres.

### 5. Módulos sueltos en la raíz del paquete

`budget_training.py`, `gru_probe.py`, `reference_probe.py` y `profiling.py` son sondas y utilidades de medida. `models/baselines` importa `gru_probe`, de modo que una sonda forma parte del camino de un modelo.

### 6. Módulos de más de 1.000 líneas

`simulation/adaptive_campaign.py` (1.568), `training/campaign_throughput.py` (1.506), `evaluation/walk_forward_comparison.py` (1.192), `observatory/collector.py` (1.170), `training/mars_titan_run.py` (1.141), `training/candidate_run.py` (1.063), `training/financial_run.py` (1.025), `training/masked_campaign.py` (1.012) y `data/macro_acquisition.py` (1.010). El tamaño por sí solo no es un defecto. Se revisará si cada uno tiene una sola responsabilidad cuando se reorganice su paquete, sin partirlos por partirlos.

### 7. Codificaciones canónicas distintas para las huellas de identidad

Hay 16 funciones `_digest`, 7 `_canonical`, 6 `_hash` y 11 `_code` repartidas por los paquetes, con al menos tres codificaciones JSON:
- `sort_keys` con los separadores por defecto (por ejemplo `data/cohort_samples.py`, `data/information_views.py`, `environments/walk_forward_receipt.py`);
- `sort_keys` con separadores compactos (los módulos de `memory`);
- `ensure_ascii=False` (`data/china_announcements.py`).

Cada identidad se calcula siempre con la misma función, así que hoy no hay huellas incoherentes. El riesgo es que dos módulos acaben calculando la huella del mismo objeto con codificaciones distintas. Además, `simulation/campaign_stage.py::_digest` no pasa `allow_nan=False` y aceptaría un `NaN` en una identidad sin fallar. Ese módulo pertenece a la PR #430, que debe corregirlo.

Las 29 copias de `_require` son triviales y unificarlas no aporta lo bastante para justificar el cambio de contenido (ver la restricción siguiente).

### 8. Restricción: módulos que firman su propio código

Unos 30 módulos incluyen `sha256(Path(__file__))` en identidades de ediciones, episodios, sesiones de memoria y ejecuciones. Por ejemplo:
- `episodes/encoding.py`, `episodes/storage.py` y `episodes/worlds.py`;
- los módulos de `memory`;
- `training/prefix_eligibility.py`, `training/predictive_inputs.py` y las ejecuciones por familia.

Mover uno de estos archivos sin tocarlo conserva su huella, porque es de contenido. Reescribir sus importaciones la cambia, y con ella todas las identidades que la incluyen. Por eso la migración tiene que hacerse antes de producir los artefactos de la campaña con la edición v3.1 y no puede tocar los módulos cuya huella quede registrada en esa edición mientras se use.

### 9. Módulos sin consumidor en el código de producción

Estos módulos solo los usan pruebas, sin importación desde `src/`, `scripts/` ni `benchmarks/` y sin punto de entrada:

| Módulo | Situación probable |
| --- | --- |
| `cm/operator_dynamics.py` | Matemática del operador C de CM-v1, pendiente de integración |
| `models/klpo_quadratic.py` | Objetivo cuadrático de KLPO para pruebas de la teoría |
| `data/cross_market.py`, `data/csi300_history.py`, `data/macro_stress_history.py` y `data/unadjusted_evidence.py` | Piezas de preparación cuyo consumidor no está en `develop` |

Ninguno debe borrarse sin revisar antes su historia. Puede que tengan resultados históricos asociados o que su integración esté en una PR abierta.

Hay además seis puntos de entrada (`if __name__ == "__main__"`) sin invocación documentada: `data/china_inventory.py`, `evaluation/financial_comparison.py`, `models/baselines/analysis.py`, `training/campaign_health.py`, `training/real_campaign.py` y `training/storage_budget.py`. Hay que documentarlos o retirarlos.

### 10. Carpetas de la raíz

- `experiments/` y `notebooks/` tienen un archivo cada una. `notebooks/` es la ubicación que fija la convención para la exploración, así que se conserva.
- `tmp/` y `build/` están ignoradas por git. `tmp/` acumula scripts de análisis y registros de sesiones anteriores fuera del control de versiones, y conviene vaciarla tras revisar que nada versionado depende de ella.
- `reports/` es la carpeta con más archivos (490) y concentra la evidencia de ingeniería e investigación. Su organización por fecha y tema es coherente y no se propone moverla.

## Estructura objetivo

Se propone solo lo que resuelve un hallazgo. No se añaden capas sin consumidor.

| Hoy | Objetivo | Hallazgo |
| --- | --- | --- |
| `training/{cohort,temporal,partition}_contract.py`, `training/prefix_eligibility.py` y `data/input_policy.py` | `contracts/` | 1 |
| `training/learning_hold.py` | `contracts/learning_hold.py` | 1 |
| `training/{run_receipts,checkpoints}.py` | `runs/` (recibos y recuperación comunes) | 1 |
| `training/corpus_*`, `*_corpus.py`, `information_inputs.py`, `predictive_inputs.py` y `target_factors.py` | `corpus/` | 1, 2 |
| `training/*_run.py` y `*_walk_forward.py` | `training/runs/` | 2 |
| `training/*_search.py`, `reference_design.py` y `selection.py` | `training/search/` | 2 |
| `training/campaign_*`, `*_campaign.py`, `storage_*`, `gpu_supervisor.py`, colas, `experiment_resources.py`, `modality_ablation_stage.py` y `cm_v1_factorial.py` | `campaign/` | 2 |
| `data/news_*`, `data/macro_*`, `data/china_*` y `data/unadjusted_*` | `data/news/`, `data/macro/`, `data/china/` y `data/unadjusted/` | 3 |
| `data/embeddings.py`, `data/embedding_placement.py` y `data/charts.py` | `data/encoding/` | 3 |
| `simulation/{market,market_rules,portfolio,environment,session_prices,reconstructed_tape,window_tapes,replay}.py` | `simulation/market/` | 4 |
| `simulation/{algorithms,training,policy_plan,adaptive_campaign,campaign_stage,campaign_receipts}.py` y `native_*` | `simulation/rl/` | 4 |
| `budget_training.py`, `gru_probe.py`, `reference_probe.py` y `profiling.py` | `profiling/` | 5 |
| funciones `_digest` y `_canonical` con codificación JSON | `contracts/identity.py`, con una codificación por versión de identidad | 7 |

`tests/` replica la misma estructura.

## Plan de migración

1. **Cuándo.** Cuando las PR abiertas que tocan cada paquete estén fusionadas y antes de lanzar la campaña. Los módulos de codificación de la edición (`data/encoding/` y `episodes/encoding.py`) se mueven antes de ejecutar la v3.1 o quedan fuera hasta terminar la campaña.
2. **Cómo.** Un commit por bloque de la tabla, con:
   - `git mv`;
   - reescritura de importaciones con una herramienta sintáctica (libcst) en lugar de sustituciones de texto;
   - actualización de enlaces en `docs/`, de órdenes `-m` en documentación y unidades systemd, y de `pyproject.toml`.

   Sin módulos de compatibilidad que reexporten las rutas antiguas, salvo que un artefacto histórico necesite importar una ruta concreta para reproducirse.
3. **Comprobación tras cada bloque.**
   - `ruff check`, `ruff format --check`, la suite completa y `check_repository.py`;
   - un recuento de dependencias que muestre que el ciclo resuelto ha desaparecido y que no aparece ninguno nuevo.
4. **Identidades.** Antes del primer bloque se registra la lista de huellas de código que cambian. Las identidades que dependan de ellas reciben versión nueva, que nunca se confunde con la anterior. Los resultados históricos conservan sus recibos con las huellas antiguas y el commit en el que se produjeron, que es lo que permite reproducirlos.
5. **Documentación.** `src/mars_titan/README.md`, `docs/engineering/architecture.md` y el README principal describen la estructura final.

## Pendiente de esta auditoría

- Revisar la historia de cada módulo del hallazgo 9 antes de proponer su retirada.
- Inventariar las configuraciones de `configs/` que ya no lee ningún código y las pruebas que solo comprueban módulos retirados.
- Revisar `native/` y `benchmarks/` con el mismo criterio.
