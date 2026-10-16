# Paralelismo de la suite rápida con pytest-xdist

Issue: [#491](https://github.com/GonxKZ/mars-titan/issues/491). Medidas del 15 y 16 de octubre de 2026 en el portátil del proyecto (16 hilos, perfil de energía de ahorro), con Python 3.12.14 y pytest-xdist 3.8.0. Ninguna ejecución aplica pasos de optimizador.

## Qué se mide

[`suite_parallelism.py`](suite_parallelism.py) ejecuta la selección de `make test-fast` (sin las marcas `slow`, `native` ni `cuda`, con `CUDA_VISIBLE_DEVICES=-1` y dos hilos por proceso) en serie y con `-n 2` y `-n 3`, repartiendo por archivo (`loadfile`) o por grupo (`loadgroup`). Cada ejecución corre en su propio trabajo `memslot suite` con el límite de 6 GiB de la plaza, y el orden de las configuraciones rota en cada una de las tres rondas.

Dentro del scope de systemd se guardan el tiempo de reloj de pytest, sin la espera de plaza, el máximo de la memoria anónima del cgroup muestreada cada segundo y los contadores de `memory.events`. `oom_kill` dice si el núcleo mató algún proceso y `max` cuántas veces se alcanzó el límite. `memory.peak` incluye la caché de página, que el núcleo deja crecer hasta el límite, así que solo se guarda como referencia. De cada ejecución se conservan también el resultado de cada prueba según JUnit y las entradas nuevas de `/tmp`. [`receipt.py`](receipt.py) compara cada ejecución terminada con la primera en serie (mismas pruebas, cada una superada, omitida o fallida igual) y resume cada configuración.

## Primera medida, con la lista de lentas atrasada

La primera medida se tomó del 15 al 16 de octubre sobre el commit bfbcadaa de la rama, que partía de `develop` en 0dfc5e2d, antes de [#635](https://github.com/GonxKZ/mars-titan/issues/635). La selección de `make test-fast` tenía entonces 8.093 pruebas superadas y 399 omitidas, porque `tests/slow_tests.txt` llevaba sin regenerarse desde #566 y dejaba dentro 409 funciones que pasan de un segundo. Su recibo es [`rounds-before-635.json`](rounds-before-635.json). Esa versión del script aún no muestreaba la memoria anónima dentro del scope, así que se tomó desde fuera cada segundo y el recibo lo indica en cada ejecución.

| Configuración | Ejecuciones terminadas | Tiempo de reloj | Memoria anónima máxima |
| --- | --- | --- | --- |
| Serie | 3 de 3 | 1.406, 1.488 y 4.049 s | 5,50 GiB |
| `-n 2` y `-n 3`, `loadfile` y `loadgroup` | 0 de 12 | | entre 5,94 y 5,96 GiB al pararse |

Las doce ejecuciones con pytest-xdist acabaron con el código 143. El núcleo mató un trabajador por memoria del cgroup y systemd paró la plaza con el resultado `oom-kill`. Llegaron a entre 6 y 29 minutos y, varias, al 89 o al 99 % de las pruebas. La serie temporal de cada proceso muestra que no se trata de un pico puntual. En `n2-loadgroup` los dos trabajadores crecieron de forma casi lineal de 0,6 a 2,8 GiB en 13 minutos, y en otras ejecuciones uno solo llegó a 4,9 GiB. En serie el único proceso de pytest llegó a 5,5 GiB, a medio GiB del límite.

Un experimento aparte con `tests/models` y `malloc_trim` tras cada prueba descarta que sea memoria retenida por el asignador de glibc: el proceso acaba con 566 MiB frente a 573 sin recortar. Lo que crece son objetos vivos de las pruebas lentas que la lista no excluía.

El tiempo de la serie varió del simple al triple. La primera ejecución empezó con una carga media de 8 a 10 por otros programas del equipo y tardó 67 minutos. Las otras dos, con carga de 4 a 6, tardaron unos 24. Esa medida descubrió el retraso de la lista, que #635 corrigió antes de repetir.

## Segunda medida, con la lista corregida

La segunda medida se tomó el 16 de octubre, de 05:40 a 06:14, sobre el commit 2eb4a83f, que ya incluye #635. La selección tiene 7.463 pruebas superadas y 377 omitidas. La carga media de un minuto estuvo entre 1,8 y 4,0 durante todas las ejecuciones. El recibo es [`rounds-after-635.json`](rounds-after-635.json).

| Configuración | Tiempos (s) | Mediana (s) | Reducción frente a la serie | Memoria anónima máxima |
| --- | --- | --- | --- | --- |
| Serie | 201,8, 205,3 y 224,9 | 205,3 | | 0,85 GiB |
| `-n 2 --dist loadfile` | 114,8, 116,1 y 117,0 | 116,1 | 1,77 veces | 1,83 GiB |
| `-n 2 --dist loadgroup` | 135,9, 136,8 y 138,5 | 136,8 | 1,50 veces | 1,95 GiB |
| `-n 3 --dist loadfile` | 91,0, 91,9 y 91,9 | 91,9 | 2,23 veces | 2,47 GiB |
| `-n 3 --dist loadgroup` | 113,1, 113,8 y 114,3 | 113,8 | 1,80 veces | 2,62 GiB |

Las quince ejecuciones terminaron con código 0, con `oom_kill` y `max` a cero, y cada prueba tuvo el mismo resultado que en la primera ejecución en serie. Ninguna dejó en `/tmp` entradas nuevas con los prefijos de tempfile, pytest o torch. Con caché de página incluida, el pico más alto fue de 3,16 GiB (`n3-loadgroup`), lejos de los 6 GiB de la plaza.

`loadfile` gana a `loadgroup` con los dos números de trabajadores. La suite no usa marcas `xdist_group`, así que `loadgroup` reparte prueba a prueba y las de un mismo archivo pueden acabar en varios trabajadores, cada uno con sus fixtures de módulo. Con `loadfile` esos fixtures se preparan una sola vez. Es la explicación más probable de que tarde más y ocupe algo más de memoria, aunque no se ha desglosado.

## Decisión

Se adopta `-n 3 --dist loadfile` en `make test-fast`. Es la configuración más rápida de las medidas, con poca dispersión entre rondas (91,0 a 91,9 s), y su pico de memoria anónima deja más de 3,5 GiB libres en la plaza. pytest-xdist 3.8.0 entra en el grupo `dev` con su dependencia execnet 2.1.2, sin cambiar ninguna otra versión del lockfile. La variable `TEST_WORKERS` del Makefile fija el número de procesos y con 0 se vuelve a un único proceso.

La suite completa (`make test`) y las pruebas lentas siguen en serie. La primera medida muestra que, con esas pruebas dentro, varios trabajadores superan el límite de la plaza. Las pruebas `cuda` tampoco entran en xdist porque comparten el candado de la GPU, y la selección rápida ya las excluye.

## Límites

- No se midieron más de tres trabajadores, que es el alcance de la issue. Con dos plazas de suite compartidas entre ramas y el perfil de ahorro de energía, tres procesos con dos hilos cada uno ya ocupan seis de los dieciséis hilos.
- La memoria anónima se muestrea cada segundo y podría no ver un pico más breve. Los contadores `max` a cero confirman que ninguna ejecución tocó el límite.
- Otros programas del equipo seguían activos. Las cifras sirven para comparar configuraciones medidas en las mismas rondas, no como tiempo de referencia de otro equipo.
- Si la lista de lentas vuelve a quedarse atrás, el pico de memoria crece con ella. La guía de pruebas explica cómo regenerarla.

## Reproducción

Desde la raíz del árbol medido, con pytest-xdist instalado en el entorno:

```bash
uv run --locked python reports/engineering/test-suite-parallelism-20261015/suite_parallelism.py SALIDA 3 serie n2-loadfile n2-loadgroup n3-loadfile n3-loadgroup
python3 reports/engineering/test-suite-parallelism-20261015/receipt.py SALIDA rounds-after-635.json
```

`SALIDA` recibe un JSON y un JUnit por ejecución y `summary.json`. El recibo deja fuera el resultado de cada prueba, que queda en los JUnit locales.
