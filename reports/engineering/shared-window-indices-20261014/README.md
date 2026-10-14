# Índices y escalas M3 compartidos por vista (14 de octubre de 2026)

Este informe acompaña a [#618](https://github.com/GonxKZ/mars-titan/issues/618). Recoge qué cambia, cómo se comprueba que los trabajos producen lo mismo y cuánto cálculo evita en el plan de A v2. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto. La protección local de aprendizaje sigue activa.

## Resumen

- Con el plan de A v2, 2.384 ajustes cronológicos construían cada uno los índices de observaciones de los cuatro tramos de su ventana en su propia carpeta. Son 51 vistas, así que 2.333 de esas construcciones repetían un índice que ya existía. Con `MARS_TITAN_SHARED_INDICES` cada índice se construye una vez por vista y lo reutilizan los demás trabajos, también los simultáneos.
- Construir los cuatro índices costó 144 s en `CN/fold-012`, 755 s en `US+CN/fold-000` y 2.181 s en `US+CN/fold-018`, en este equipo y con carga ajena. Las 19 vistas conjuntas tienen 112 constructores cada una, y ahí está casi todo el ahorro: entre 442 y 1.278 horas de cálculo evitadas solo en esas vistas.
- Las escalas de M3 se estiman una vez por ventana en lugar de cuatro. Son 57 recorridos completos del tramo de ajuste de US+CN menos.
- Un trabajo con índices compartidos produce los mismos bits que con índices propios. Las 8 mutaciones dirigidas del bloqueo, la clave, la liberación y las escalas hacen fallar alguna prueba.
- El pico de disco estimado del recorrido por ventanas con la retención v3 pasa de 92,57 GB a 93,04 GB, porque ahora cuenta los índices de cada vista durante todas las fases de su ventana.

## Qué cambia

`training/window_indices.py` decide dónde vive cada índice. Sin la variable, cada trabajo los guarda en `indices` de su carpeta, como antes. Con ella, los índices de una vista van a una carpeta común con la huella de la vista, y dentro hay un índice por tramo cuyo nombre lleva la huella de su identidad completa (`index_identity`): vista, tramo con su calentamiento, receta, versiones de PyArrow, NumPy y DuckDB y huella del código que lo construye. Un cerrojo con espera junto a cada índice impide que dos trabajos lo construyan a la vez. El segundo espera y lo reutiliza, y `prepare_observation_index` comprueba la identidad al reutilizarlo y completa un índice que quedó a medias.

La identidad completa en el nombre resuelve un caso que el nombre por vista y tramo no cubría. Si el código o una biblioteca cambian entre un corte y la reanudación, el índice nuevo va a otra carpeta en lugar de chocar con el anterior, que se queda sin usar hasta que se libera la vista.

Todas las familias con índice usan la misma regla: Titans-MAC, núcleos y lectores de CM-v1, lectores de MARS-TITAN, B6, control en línea, GRU candidata, adaptadores cronológicos y regeneraciones. Las escalas de M3 se guardan junto a los índices de la vista, con un nombre que lleva la huella del índice de ajuste, del bloque, de la muestra, de la semilla y del código que las estima. Al leerlas se comprueba que proceden del mismo índice y de la misma vista.

La campaña fija la raíz en `shared-indices` de su salida, salvo que un proceso superior ya la haya fijado, y borra los índices de una vista al confirmar el último trabajo de su ventana. En el recorrido por ventanas la fija el propio recorrido, y los índices se borran al final de la liberación de cada ventana, después de las regeneraciones que los reutilizan. El borrado empieza con un renombrado atómico, así que un corte nunca deja un índice a medias con su manifiesto.

## Comprobaciones

| Comprobación | Prueba |
| --- | --- |
| Un trabajo de Titans-MAC con índices compartidos escribe los mismos informes, tablas por fila y estados que con índices propios | `tests/training/test_titans_walk_forward.py::test_shared_indices_keep_the_window_bit_for_bit` |
| Los trabajos de una vista construyen cada tramo una vez y el índice compartido tiene la misma identidad que uno propio | `tests/training/test_window_indices.py` |
| Dos trabajos simultáneos esperan al primero y reutilizan su índice | `test_simultaneous_jobs_wait_for_the_first_build_and_reuse_it` |
| Un índice a medias se completa y un cambio de código construye otro al lado | `test_an_index_left_half_built_is_completed_with_the_same_identity`, `test_a_code_change_before_resuming_builds_new_indices_beside_the_old_ones` |
| Un corte durante el borrado no deja un índice a medias, tampoco dos cortes seguidos | `test_a_cut_while_releasing_never_leaves_a_half_deleted_index` |
| Las escalas de M3 se estiman una vez y se leen con los mismos bits | `test_m3_scalers_of_a_window_are_estimated_once_and_read_with_the_same_bits` |
| La campaña reducida comparte los índices y los borra al cerrar cada ventana | `tests/training/test_titans_campaign.py::test_jobs_share_the_indices_of_each_view_and_the_campaign_releases_them` |
| La retención v3 borra los índices de la ventana después de regenerar, y las regeneraciones aún los encuentran | `tests/posttraining/test_retention_v3.py::test_v3_releases_the_adapter_rows_by_blocks_and_publishes_the_same_comparison` |
| Las estimaciones de disco cuentan cada vista una sola vez y durante toda su ventana | `test_campaign_storage.py`, `test_storage_budget.py` y `test_rolling_storage.py` |

Las mutaciones están en [`mutations.py`](mutations.py) y su resultado en [`mutations.json`](mutations.json). Se aplican una a una sobre una copia del árbol y todas hacen fallar alguna prueba: clave sin el código, construcción sin cerrojo, renombrado antes de limpiar restos, borrado sin renombrado, escalas sin el código, escalas sin comprobar su origen, borrado de otras vistas y liberación antes de regenerar.

## Coste evitado

[`index_builds.py`](index_builds.py) cuenta en el plan de A v2 los trabajos que construyen índices. De los 4.160 trabajos con índice, 1.776 son continuaciones de una meseta que reanudan en su carpeta y reutilizan sus índices. Quedan 2.384 ajustes que construyen los cuatro tramos: 8 por vista en US y en CN y 112 por vista en US+CN ([`index-builds.json`](index-builds.json)).

[`index_costs.py`](index_costs.py) abre una vista y construye sus cuatro índices con el código de `develop` (06e69db6), como el primer trabajo de la ventana. Las medidas se hicieron en este equipo (16 hilos, 32 GB) con otras cargas en marcha, que se anotan. Las tres están en [`index-costs.json`](index-costs.json).

| Vista | Filas de ajuste | Ajuste | Validación | Calibración | Evaluación | Total | Carga media |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CN/fold-012` | 4.908.804 | 64 s | 27 s | 26 s | 27 s | 144 s | sin anotar |
| `US+CN/fold-000` | 2.723.441 | 236 s | 185 s | 175 s | 160 s | 755 s | entre 4 y 8 |
| `US+CN/fold-018` | 32.723.183 | 368 s | 387 s | 666 s | 760 s | 2.181 s | entre 8 y 10, con la batería de pruebas |

En `fold-018`, los tramos de validación, calibración y evaluación tienen entre 9 y 18 veces menos filas que el de ajuste y cuestan lo mismo o más. La construcción recorre las fechas de todas las muestras de cada activo en cada tramo, así que su coste depende sobre todo del número de archivos y grupos de la vista y poco de las filas del tramo.

Con índices compartidos, cada vista construye sus cuatro índices una vez. En `CN/fold-012` se evitan 7 construcciones, unos 17 minutos. En `US+CN/fold-018` se evitan 111, unas 67 horas de cálculo con la carga de la medida. Las vistas conjuntas crecen con la ventana, así que sus 19 × 111 = 2.109 construcciones evitadas cuestan entre 2.109 veces el coste de `fold-000` y 2.109 veces el de `fold-018`: entre 442 y 1.278 horas con la carga de cada medida. No se han medido las demás vistas y esta cota no incluye las 32 vistas de un solo mercado, cuyo ahorro es mucho menor.

Las escalas de M3 dependen solo del tramo de ajuste. En cada ventana conjunta había 4 ajustes M3 con meseta (dos casos de búsqueda y dos finalistas) que las estimaban recorriendo el tramo completo con el lector. Las continuaciones ya las leían de su meseta. Ahora se estiman una vez por ventana, así que se evitan 3 × 19 = 57 recorridos. No se ha vuelto a medir su duración. La medida anterior de `benchmarks/chronological_replay_patches.py` la situaba por encima de 40 minutos por recorrido en US+CN con el lector de entonces.

## Disco

La estimación de A v2 con la retención v3 (`run_masked_campaign.py storage --schedule --adapter-blocks`, conteos de `campaign-storage-a-v2-v31-20261014`) cambia así:

| Escenario del recorrido por ventanas | `develop` 06e69db6 | Con #618 |
| --- | --- | --- |
| Todas las regeneraciones idénticas | 92,57 GB | 93,04 GB |
| Solo los adaptadores regenerados | 242,13 GB | 242,61 GB |
| Ninguna regeneración idéntica | 863,07 GB | 863,54 GB |

El pico sigue en la ablación de `fold-018`. La diferencia de 0,47 GB son los índices compartidos de esa vista, que ahora se cuentan en todas las fases de su ventana junto con los índices escalonados de los adaptadores que ajustan. En el plan de la campaña y en su presupuesto, los índices de una vista ocupan sitio desde su primer trabajo pendiente hasta su último trabajo con índice, una sola vez. Con 131 GB libres el 14 de octubre, el escenario de referencia sigue cabiendo.

## Límites

Las horas son tiempo de construcción medido con carga ajena en dos vistas conjuntas y una de un mercado, no una proyección completa de la campaña. Un trabajo que espera el cerrojo de otro no construye, pero su plaza espera hasta que el índice existe. El ahorro de las escalas M3 se deduce del recuento y de una medida anterior.
