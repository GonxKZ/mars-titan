# Contrastes secundarios de capacidad predictiva

Comprobaciones técnicas de [#493](https://github.com/GonxKZ/mars-titan/issues/493) hechas el
10 de octubre de 2026 en CPU y sin ningún paso de optimizador. El coste se midió sobre el
commit `903e4fc9`. La importación, la regresión sin la sección y la mutación se repitieron
sobre `557254be`, que solo lleva las importaciones de arch y statsmodels al interior de las
funciones que calculan y comprueba al empezar la evaluación que están instaladas.
Ninguna usa predicciones de la campaña ni datos de mercado. La sección `predictive_ability` y
sus reglas están en las [métricas](../../../docs/research/metrics.md#contrastes-secundarios-de-capacidad-predictiva).

## Entorno

Python 3.12.14, NumPy 2.5.3, arch 8.0.0 y statsmodels 0.15.0, resueltos con
`exclude-newer = "2026-09-26T23:59:59Z"`. AMD Ryzen 9 8945HS de 16 hilos compartida con otras
sesiones, dos hilos para BLAS y OpenMP y `CUDA_VISIBLE_DEVICES=-1`.

## Coste

[`cost.json`](cost.json) sale de
[`benchmarks/predictive_ability_cost.py`](../../../benchmarks/predictive_ability_cost.py) con
`--repeats 2`. Usa los calendarios reales de US (2005 a 2023, 4.781 días) y de CN (2011 a 2023,
3.159 días), que dan 4.887 días UTC en la vista conjunta, los 24 brazos y las 13 familias del
ámbito US+CN de la comparación conjunta declarada, 2.000 réplicas y bloques de 16 días. Las
pérdidas son sintéticas con semilla fija y solo fijan las formas.

| Medida | Primera repetición | Segunda repetición |
| --- | --- | --- |
| Sección completa, tres vistas | 29,2 s | 33,1 s |
| SPA y StepM | 12,7 s | 14,1 s |
| MCS | 12,3 s | 14,4 s |
| Diebold-Mariano | 0,05 s | 0,06 s |
| Longitud de Politis y White | 0,08 s | 0,09 s |

La carga media al empezar era de 8,4 en el último minuto y 11,8 en los últimos quince, así
que la diferencia entre repeticiones es sobre todo carga concurrente. Una medida anterior con
el mismo árbol y carga 16,2 dio 34 y 49 s. El pico de memoria reservada por NumPy, medido aparte con `tracemalloc`, fue
de 107 MiB. Las dos repeticiones dieron un informe con la misma huella. No se midieron la GPU,
que estos contrastes no usan, ni la energía.

## Importación sin el extra research

La declaración de la sección y sus constantes solo usan NumPy. `campaign_plan`, `policy_plan`,
`simulation.campaign_stage` y `walk_forward_comparison` se cargan sin arch ni statsmodels, como
en el entorno de `rl check` (`--extra cuda --extra reinforcement`).
`test_campaigns_load_and_rl_check_runs_without_arch_or_statsmodels` lo comprueba en un
subproceso que bloquea los dos paquetes: carga las campañas A y A v2, ejecuta `rl check` y
no registra ningún intento de importarlos. La evaluación que declara la sección sí falla, con
el nombre del paquete, antes de leer ninguna fuente.

Importar `campaign_plan` y `walk_forward_comparison` en un proceso nuevo tardó 1,94 s de
mediana con `557254be` (1,86 a 2,17 s) y 3,43 s con `ddefb48a` (3,01 a 3,82 s), en cinco
repeticiones alternas de cada versión con una carga media de 26 a 28.

## Paridad del bootstrap

[`test_arch_bootstrap_parity.py`](../../../tests/evaluation/test_arch_bootstrap_parity.py)
comprueba que `circular_block_indices` y `CircularBlockBootstrap` de arch, con el mismo
`numpy.random.Generator`, dan los mismos índices en seis formas, entre ellas bloques que no
dividen a los días, un bloque que cubre toda la serie y bloques de un día. También que sortear
por tandas conserva el flujo y que el MCS de arch remuestrea los mismos días que el contraste
principal. Las variantes con otro inicio, otra longitud o sin recorrido circular dan índices
distintos.

## Sin cambios sin la sección

La sección es opcional. Con el código de `develop` (`b92e07a0`) y el de la rama (`557254be`) se
evaluaron los mismos estudios sintéticos US+CN, US y CN sin la sección. Los informes, salvo la
fecha, los recursos y las huellas del código, y las tablas `sessions.parquet` resultaron
idénticos.

## Mutación dirigida

[`mutations.json`](mutations.json) recoge 27 defectos aplicados de uno en uno sobre una copia
del árbol de `557254be` y ejecutados contra `test_predictive_ability.py`,
`test_arch_bootstrap_parity.py` y `test_walk_forward_predictive_ability.py`. Fallan 26, entre
ellos importar arch o statsmodels al cargar el módulo, no comprobar las bibliotecas al empezar
la evaluación y comprobarlas después de leer las fuentes. Sobrevive `spa_studentize_true`, que
es equivalente con arch 8.0.0, porque esa versión ignora `studentize` en el SPA. Una prueba
fija ese comportamiento para detectar cuándo cambia.
