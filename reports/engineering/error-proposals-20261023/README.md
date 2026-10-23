# Controles finitos de propuestas sobre errores maduros

El [commit e266a1398](https://github.com/GonxKZ/mars-titan/commit/e266a13986488f2d8ad23c4a27566e805a3456bc) implementa un plan temporal, una lectura completa de la fase cerrada y cuatro controles programados de propuestas. La [especificación de las API](../../../docs/engineering/error-proposals.md) describe entradas, cortes, reservas y recuperación. Este incremento no entrena un proponente, no selecciona consumidores y no mide una mejora predictiva.

Las ventanas quedan acotadas a meses o, como máximo, 366 días entre generación y evaluación. La fuente de generación pertenece a entrenamiento o validación y conserva todas sus emisiones. El test de 2024 y la reanudación automática permanecieron cerrados. No se entrenó con el histórico completo.

## Comprobaciones ejecutadas

La verificación final pasó 252 casos únicos de `tests/proposals/`, sin fallos, errores u omisiones. Duró 89,965 s como proceso y 80,73 s dentro de pytest. Incluye los 80 casos anteriores de registro, revisión y persistencia. Los eventos observados de optimizador fueron cero. Las fixtures nativas se ejecutaron en `cuda:0` con el binario existente identificado en el [resumen verificable](summary.json). No se atribuye una compilación nueva.

La revisión adversarial detectó dos fallos corregidos antes de la verificación final. Un marco de muestras contradictorio podía acompañar a una revisión intacta. También podía consumirse un intento antes de detectar que padre, contrato o corte no correspondían al plan. Las pruebas actuales rechazan ambos casos antes de reservar. La recuperación de un fallo posterior a la reserva conserva el cargo pendiente.

Una prueba numérica adicional reprodujo la pérdida de un término al normalizar pesos extremos antes de multiplicar. Con peso CN `1e308`, peso US `1e-308`, pérdida CN cero y pérdida US `1e308`, el resultado anterior era cero y el resultado representable es `1e-308`. La reducción final entre uno o dos mercados ahora opera sobre fracciones exactas de los valores recibidos y redondea una vez a float. No convierte las entradas originales en medidas exactas del mercado.

Se detectaron tres mutaciones dirigidas: incluir el empate en el disparador, omitir el vínculo con el registro y omitir la validación del marco. Cada una pasó la referencia G1, produjo el fallo semántico previsto en CALL con M1 y volvió a pasar tras restaurar en G2. No hubo errores de colección, omisiones ni intentos de optimizador. Los tiempos de estas tres secuencias están separados de los 252 casos únicos. Tres mutantes detectados no acreditan resistencia universal a mutaciones.

Coverage 7.16.1 y Radon 6.0.1 registran 501/509 sentencias y 69/72 ramas en los cuatro archivos modificados de producción. El CRAP máximo es 28. Se usa `CRAP = v²(1-c)³ + v`, con complejidad Radon y cobertura de sentencias dentro del rango de cada función. Las ramas se informan aparte. Las coberturas proceden de dos mediciones sobre archivos disjuntos, con sus SHA finales: tres archivos en la pasada instrumentada de 251 casos y `error_rule.py` en la de 91 casos posterior a la corrección numérica. Esas repeticiones no añaden casos únicos ni prueban ausencia de defectos.

Las comprobaciones de Ruff y formato pasaron para `src/mars_titan/proposals` y `tests/proposals`. Las pruebas de comportamiento se pueden repetir con el entorno del proyecto y `MARS_TITAN_EPISODIC_NATIVE` apuntando al enlace nativo compatible:

```bash
MARS_TITAN_REQUIRE_CUDA=1 uv run --no-sync pytest tests/proposals -q
uv run --no-sync ruff check src/mars_titan/proposals tests/proposals
uv run --no-sync ruff format --check src/mars_titan/proposals tests/proposals
```

## Coste del recorrido de propuestas

El perfil reutilizó una fase nativa cerrada con seis eventos, dos cohortes, ocho emisiones sintéticas y cuatro flujos US. El registro contenía dos recetas y una candidata. Cada control tuvo un calentamiento y siete repeticiones con ledger nuevo, 32 invocaciones en total. Se midió la lectura y verificación, la creación del ledger, la propuesta y la publicación durable, con sincronización CUDA antes y después. No se midió el forward del predictor ni se emitieron nuevas predicciones durante ese perfil.

| Control | Mediana observada | Mínimo | Máximo |
| --- | ---: | ---: | ---: |
| `none` | 241,176 ms | 202,288 ms | 517,848 ms |
| `fixed` | 228,128 ms | 227,486 ms | 231,607 ms |
| `enumerate` | 599,396 ms | 349,249 ms | 737,022 ms |
| `mature_parent_vs_zero` | 746,366 ms | 269,182 ms | 830,995 ms |

Los controles se midieron en bloques, sin aleatorizar el orden, con procesos Java y un servidor de modelo local concurrentes. La enumeración solo tenía una candidata. Las diferencias de esa tabla no demuestran una ventaja entre controles ni describen el coste de enumerar siete candidatas. Con siete muestras, los p95 y p99 empíricos del resumen coinciden con el máximo observado. No estiman con precisión las colas de producción.

El observador de llamadas Python estuvo activo durante calentamientos y guardas y desactivado en las muestras cronometradas. En el ámbito observado no hubo constructores o forwards de modelos ni intentos de optimizador. Las fuentes, el archivo, el estado del padre y los RNG conservaron sus identidades al terminar. Los recibos completos y el ejecutor local del perfil se conservan fuera de Git, identificados por sus huellas. Repetir exactamente ese perfil requiere ese ejecutor, además de la fixture versionada.

La RTX 4070 Laptop mantuvo el padre residente. En los cuatro controles, el allocator Torch conservó 67.336.704 bytes asignados y 69.206.016 reservados, sin incremento observado durante las siete muestras. Esto no implica transferencias CPU/GPU nulas ni incluye toda la memoria del dispositivo. El máximo acumulado de RAM del proceso pytest fue 1.344.864.256 bytes e incluye la fixture y el entorno. No se atribuye ese valor exclusivamente al control.

Cada lectura contabilizó 57.073 bytes lógicos de fuente. El `state.json` retenido ocupó 15.186 bytes para `none` y 16.415 para cada control activo. Son tamaños lógicos, no tráfico físico de almacenamiento. El perfil duró 26,032 s dentro del ejecutor y 52,651 s como proceso completo. No se midieron energía, transferencias, lanzamientos de kernels ni RSS instantánea.

El coste corresponde a una operación posterior al cierre de fase. No se añade al forward de cada activo. La utilidad financiera de las candidatas, el aprendizaje del proponente y la comparación de sus trayectorias posteriores siguen sin evaluar.
