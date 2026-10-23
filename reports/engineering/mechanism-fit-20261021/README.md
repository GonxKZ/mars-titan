# Máscara de acciones y preparación del ajuste protegido

Fecha: 21 de octubre de 2026. Código: `d0768ad43e6ea5b17625c4508c71e9d94ef767ad`.

La política incorpora una máscara opcional para impedir que una acción gane cuando sus objetivos requeridos no tienen datos suficientes. La entrada de ajuste de sus ganancias está preparada bajo protección. No se han leído fuentes para ajustarla ni ejecutado normalización o regresiones. No se han obtenido coeficientes aprendidos ni medido una mejora predictiva.

| Componente | Estado comprobado |
| --- | --- |
| Máscara y checkpoint anterior sin ella | Pruebas CPU y CUDA con coeficientes manuales, en FP32 y FP64. |
| Recibo declarado de ajuste | Referencia conservada sin afirmar que el aprendizaje esté verificado. |
| Sesión nativa FP32 | Selección, propuestas, recepción de etiquetas y recuperación comprobadas en CPU y CUDA. |
| Contrato y protección del ajuste | Validación y rechazo antes de inspeccionar configuración o fuentes. |
| Lector positivo, normalización y solver del ajuste | Código preparado. Ejecución y validación numérica pendientes. |
| Selección temporal y utilidad predictiva | Pendientes. Campaña completa y test final cerrados. |

El [contrato técnico](../../../docs/engineering/mechanism-policy-fitting.md) describe las 12 ganancias puntuales compartidas entre calibraciones, los 36 objetivos de intervalos, los pesos y las ausencias. Solo admite sesiones enteramente cerradas dentro del tramo declarado. Esa restricción no resuelve todavía la lectura de un archivo más largo con recepciones de distintos tramos. La [auditoría de fuentes](source_audit.md) separa el argumento algebraico del rango, la API consultada y lo que se ha ejecutado.

## Pruebas y límites de la evidencia

La [pasada final](verification.json) registra 402 casos aprobados, 94 marcados CUDA, sin fallos, errores ni omisiones. Las tres baterías nuevas aportan 54 casos, cinco marcados CUDA. Incluyen máscara antes de argmax, scores conservados, aislamiento de los buffers, RNG, checkpoint anterior, rechazo de escalares inválidos, límites temporales, presupuestos, recibo declarado y sesión FP32 recuperable. Es una selección de integración y regresión, no toda la suite del proyecto.

Se reprodujo un defecto en `MechanismPolicy.validate()`: aceptaba escalas o referencias inválidas cambiadas después de construir el objeto y antes de congelarlo. Cinco pruebas fallaron antes de añadir la revalidación. La comprobación ampliada se repitió después de corregirlo. El inventario del bloqueo incluye las dos entradas nuevas y una guarda anterior de refinamientos que ya existía pero carecía de referencia en ese inventario.

La [mutación dirigida](mutations.json) detecta 14 cambios seleccionados, con una base inicial y otra restaurada de 49 casos CPU cada una. Los mutantes de las guardas intentan inspeccionar objetos preparados para fallar antes de cualquier aprendizaje. No se retiró la protección ni se ejecutaron regresiones para comprobarlos. Se dejó fuera un mutante equivalente: eliminar la primera comprobación de un SHA sigue siendo rechazado por su revalidación posterior. No se presenta esta selección como un porcentaje global de mutación.

La [medida de calidad](quality.json) usa coverage.py 7.16.1 con ramas y Radon 6.0.1. Comprende 36 funciones de cinco módulos, con 549 de 700 sentencias y 206 de 262 ramas ejecutadas. La política alcanza 111 de 112 sentencias y 39 de 40 ramas. El contrato alcanza 52 de 54 y 16 de 18. El módulo de ajuste conserva solo 18 de 129 sentencias ejecutadas y ninguna de sus 16 ramas: la protección impide recorrer el lector y el solver. Sus mayores CRAP son 401,82 en `fit_policy` y 321,33 en `_cohorts`. Estos valores señalan una validación pendiente, no se ocultan con pruebas simuladas ni se interpretan como una implementación científicamente completa.

Las primeras ejecuciones de cobertura por nombre de módulo no se usan como evidencia final. La medición seleccionada utiliza directorios y mantiene incluidas las rutas sin ejecutar del ajuste. Los intentos y recibos anteriores están identificados en el [índice externo](external-artifacts.json). Ruff y el formato pasan. No se han añadido dependencias o kernels ni modificado C++.

Para repetir la selección en el entorno fijado, con CUDA y el enlace nativo configurados:

```bash
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_REQUIRE_NATIVE=1 MARS_TITAN_REQUIRE_CUDA=1 uv run --no-sync pytest -q tests/models/titans/test_mechanism_action_support.py tests/models/titans/test_mechanism_policy.py tests/models/titans/test_mechanism_consumer.py tests/memory/test_mechanism_action_support_session.py tests/memory/test_native_mechanism_collection.py tests/memory/test_native_collection_guard.py tests/memory/test_financial_mechanisms.py tests/memory/test_mechanism_selection.py tests/memory/test_mechanism_collection.py tests/memory/test_mechanism_supervision.py tests/training/test_mechanism_policy_fit.py tests/training/test_learning_hold_inventory.py tests/training/test_refinement_policy_fit.py::test_fit_requires_authorization_before_inspecting_any_arrays
```

## Coste del selector con coeficientes manuales

La medida usa una RTX 4070 Laptop, `cuda:0`, dos hilos CPU, 21 rasgos y una o dos cohortes por decisión. Compara la ruta anterior, una máscara que habilita todo y otra que excluye la acción 4. Hay diez llamadas de calentamiento y cien medidas por modo, precisión y tamaño, con orden rotatorio. Cada llamada se sincroniza explícitamente. Los intervalos entre eventos CUDA incluyen las operaciones del stream y sus esperas entre lanzamientos, no son un perfil exclusivo de kernels.

| Precisión | Cohortes | Ruta anterior, mediana | Máscara completa, mediana | Coste adicional observado |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1 | 0,54137 ms | 0,55683 ms | 15,46 µs |
| FP32 | 2 | 0,53736 ms | 0,55025 ms | 12,88 µs |
| FP64 | 1 | 0,55615 ms | 0,56638 ms | 10,23 µs |
| FP64 | 2 | 0,57125 ms | 0,58436 ms | 13,11 µs |

Los scores coinciden bit a bit entre los tres modos. La máscara completa conserva la elección anterior. La parcial elige la acción 3 y conserva fuera de la admisibilidad la acción 4, aunque su score sea mayor. El RNG no cambia. El pico del asignador aumenta 512 bytes en esta comparación, con los tres controles residentes. No equivale al tamaño exclusivo de la política ni a toda la VRAM.

Este coste corresponde solo al forward del selector. No incluye ajuste, codificadores, MAC, consultas al banco, escritura nativa ni el predictor financiero completo. No se midió energía ni se ejecutaron otras cargas propias durante la medida. No se detuvieron aplicaciones ajenas. Los percentiles altos de cien llamadas son descriptivos de este ensayo local.

El [protocolo](cost-protocol.json), el [coste](cost.json), la [comparación](comparison.json) y el [recibo del proceso](process-cost.json) conservan versiones, parámetros y hashes. Las muestras completas permanecen fuera de Git. Para repetir la medida, con una salida nueva:

```bash
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python reports/engineering/mechanism-fit-20261021/benchmark.py --output /tmp/mars-titan-action-support-cost.json
```

El ajuste positivo, su recuperación y publicación con un recibo verificado, la selección temporal y la evaluación de una política autónoma siguen pendientes. La máscara y un hash declarado no acreditan que el modelo haya aprendido a elegir mejores mecanismos.
