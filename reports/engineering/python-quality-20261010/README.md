# Cobertura, complejidad y CRAP comunes en Python

Fecha: 10 de octubre de 2026. Issue [#489](https://github.com/GonxKZ/mars-titan/issues/489).

Hasta ahora cada informe calculaba la complejidad y CRAP de sus fuentes Python con órdenes propias fuera del repositorio. Unos usaban Radon 6.0.1, dos usaron Lizard 1.24.0 y ninguno dejaba un guion que se pudiera volver a ejecutar. `scripts/summarize_python_quality.py` reúne ese cálculo. Lee el JSON de coverage.py y calcula la complejidad de cada función sobre la misma región en la que coverage.py cuenta sus sentencias. Es el equivalente Python de `scripts/summarize_native_quality.py`, que sigue con Lizard y la cobertura de LLVM para el código nativo.

## Convenciones

| Elemento | Convención |
| --- | --- |
| Regiones | Las de `coverage.regions.code_regions` en coverage.py 7.16.1, la función que coverage.py usa para repartir sentencias. Una función anidada es su propia región. Una función definida en un `else`, un `except` o un `finally` no es región para coverage.py y sus sentencias cuentan en la región que la contiene. |
| Complejidad | `ComplexityVisitor` de Radon 6.0.1 aplicado a cada función, la misma herramienta de los recibos anteriores. Suman uno `if`, `elif`, la expresión condicional, `for`, `while` y su `else`, cada `except`, el `else` de un `try`, `assert`, cada operando booleano después del primero y cada cláusula `for` o `if` de una comprensión. Un `match` suma sus casos menos uno si alguno captura sin patrón. No suman `with`, `finally`, los manejadores de `except*`, las funciones y clases anidadas ni la condición de un `assert`. |
| Cobertura | Sentencias ejecutadas entre sentencias de la región. Las ramas se publican aparte y no entran en CRAP. |
| CRAP | C² × (1 − cobertura)³ + C. Una función sin sentencias medibles conserva C. |

El resumen se detiene si la fuente cambió después de la medida (la línea de una región o las líneas de sentencia no coinciden con las medidas), si coverage.py no registra una región, si el JSON no es del formato 3 o si se piden ramas y la medida no las trae. Si dos funciones comparten nombre cualificado en un archivo, como un `setter` de propiedad, coverage.py solo mide la última y las anteriores aparecen en `shadowed_regions` sin CRAP.

Se eligieron las reglas de Radon y no las de ruff (`C901`) para que los recibos anteriores sigan siendo comparables. ruff aplica el McCabe de `mccabe`: no cuenta operandos booleanos, expresiones condicionales, comprensiones, el `else` de los bucles ni `assert`, y suma las funciones y clases anidadas en la función que las contiene.

## Continuidad con los recibos anteriores

`radon_receipts.py` recorre los JSON de `reports/`, toma cada función Python con su complejidad registrada y la recalcula sobre la fuente del commit más antiguo que contiene el contenido actual del recibo. No sirve el último commit que lo toca, porque las publicaciones en `main` vuelven a escribir todos los recibos junto con fuentes posteriores a su medida. El resultado está en [`radon-receipts.json`](radon-receipts.json).

| Estado | Funciones | Explicación |
| --- | ---: | --- |
| Iguales | 664 | 648 de 24 recibos con Radon y 16 de los dos recibos del 4 de octubre con Lizard |
| Distintas | 4 | La fuente medida no es la del recibo. Ver la tabla siguiente |
| Ambiguas | 4 | `__init__` sin línea en `historical-encoding-budgets-20261008.json`. Las dos candidatas valen 3 y 12 y el recibo registra 3, 3, 12 y 12 |
| No encontradas | 2 | Filas de clase (`InitialPolicyCache` y `_Reader`) en `campaign-comparison-20261005-quality.json`. Radon da complejidad a clases y el resumen solo mide funciones |

En total son 674 funciones de 26 recibos. Siete filas nombraban solo el archivo (`macro_h15_archive.py` y `cohort_samples.py`) y se resolvieron porque ese nombre era único en la revisión del recibo.

| Recibo | Función | Recibo | Ahora | Causa |
| --- | --- | ---: | ---: | --- |
| `campaign-observatory-quality.json` | `collector.public_run` | 40 | 53 | El archivo se creó en el mismo commit que el recibo, que se midió sobre una versión anterior sin registrar su huella |
| `campaign-observatory-quality.json` | `collector.validate_record` | 21 | 34 | La misma causa |
| `episodic-session-20261008.json` | `retention_bank._check_receipt` | 23 | 35 | La huella registrada en el recibo no es la de la fuente de ese commit |
| `episodic-session-20261008.json` | `session_artifacts.prune_unreferenced` | 7 | 8 | La misma causa |

Las 16 funciones de Lizard coinciden porque no usan ninguna construcción en la que las dos convenciones difieran. No demuestran que Lizard y Radon den lo mismo en general.

## Pruebas y mutación

Radon 6.0.1 se declara en la cabecera del guion, como Lizard en el resumen nativo, y en el grupo de desarrollo junto a coverage.py 7.16.1. El recibo registra las dos versiones. `tests/tooling/test_python_quality.py` fija cada regla de Radon con casos explícitos (incluidos los patrones de `match` con captura, alternativas y `as`), así que otra versión que cambie una regla rompe la prueba. También comprueba las regiones de coverage.py, la fórmula de CRAP, que solo se acepten funciones, los errores ante una fuente cambiada, una región ausente o una medida sin ramas, y la orden completa.

Son 35 pruebas. La campaña dirigida de 18 mutaciones del guion (exponente y términos de CRAP, complejidad con las funciones anidadas o sin `assert`, entrada que no es función, versión de Radon sin registrar, regiones de clase, comprobación de línea, regiones ausentes o repetidas, funciones vacías, formato, ramas, selección, orden y recorte) terminó con las 18 detectadas ([`mutations.json`](mutations.json)). Se descartó un mutante equivalente: `total_complexity` del visitante coincide con la complejidad de la función cuando se le pasa un único nodo, como se comprobó en las 13.242 funciones de `src/mars_titan`, `scripts` y `tests`.

## Un recibo recalculado de principio a fin

Se repitió la medida de `named-benchmarks-verification-20261009.json` con coverage.py 7.16.1 y ramas (56 pruebas pasadas y 5 deseleccionadas, como el recibo) y se resumió con el guion. Las siete funciones de `simulation/benchmark_catalog.py` dan complejidades 14, 9, 5, 4, 3, 3 y 2, con 86 de 86 sentencias y 36 de 36 ramas. Son los mismos valores del recibo.

## Límites

- CRAP combina complejidad y recorridos observados. No demuestra ausencia de errores y no se fija ningún umbral.
- La cobertura de sentencias no distingue qué operandos de una condición se evaluaron, aunque la complejidad sí los cuenta.
- Como en Radon 6.0.1, los manejadores de `except*` y la condición de un `assert` no suman, y el cuerpo de una clase definida dentro de una función no suma en ella aunque coverage.py cuente allí sus sentencias.
- La fuente se contrasta con la medida por sus regiones y sus líneas de sentencia, porque el JSON de coverage.py no guarda la huella de lo que midió. Un cambio que las conserva, como otra condición en la misma línea, no se detecta, así que el resumen se ejecuta sobre el mismo árbol que se midió.
- Los recibos anteriores no se reescriben. Conservan la convención y la herramienta que declararon.

## Reproducción

Desde la raíz del repositorio, con las herramientas del grupo de desarrollo:

```bash
uv run python reports/engineering/python-quality-20261010/radon_receipts.py \
  --output reports/engineering/python-quality-20261010/radon-receipts.json
uv run pytest tests/tooling/test_python_quality.py
```
