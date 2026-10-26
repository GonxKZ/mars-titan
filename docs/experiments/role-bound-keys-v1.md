# Claves ligadas al régimen observable

`role_bound_keys_v1` (B3, [#750](https://github.com/GonxKZ/mars-titan/issues/750)) es un factor opcional de la corrección madura. Transforma la clave del codec antes de leer y escribir B6, B1 o B2. No cambia el padre Titans-MAC, su memoria neuronal ni el objetivo residual. Permanece desactivado en las configuraciones canónicas.

La etiqueta B3 pertenece al catálogo de este estudio. No modifica el identificador B3 que el catálogo histórico utiliza para la referencia GRU. La identidad de esta variante es `role_bound_keys_v1`.

El contrato está en [role_binding.py](../../src/mars_titan/memory/role_binding.py), se conecta mediante `MatureCorrection` en [associative_memory.py](../../src/mars_titan/memory/associative_memory.py) y tiene [configuración independiente](../../configs/experiments/role-bound-keys-v1.json). Reutiliza `CorrectionInference` y `FinancialSession`.

## Operador y controles

Sea x la clave del codec, normalizada en L2 y FP64, y r su ruta observable en la decisión. Se generan cinco códigos cᵣ de 64 signos, cada uno en {−1, +1}, con un PCG64 privado y semilla declarada. El dominio del generador es `[0xB3, 0, seed]`, distinto del usado por las áreas y por el barajado de correspondencias. B3 calcula:

```text
b(x, r) = x ⊙ cᵣ
b⁻¹(y, r) = y ⊙ cᵣ
```

Como cᵣ ⊙ cᵣ = 1, aplicar dos veces el producto recupera los bits de una clave finita, incluidos los ceros con signo y los subnormales en las condiciones comprobadas. Conserva la norma y los productos escalares entre claves ligadas al mismo rol. Roles distintos cambian la geometría compartida, pero no se presupone que sus códigos sean ortogonales en 64 coordenadas.

El antecedente es el producto bipolar descrito por Kanerva (2009), páginas 147 y 156, que relaciona con HRR complejo restringido a los valores 1 y −1. Esta implementación no es un backend complejo FHRR general ni una implementación identificada como HLB. La fuente no demuestra un beneficio financiero. [Artículo y DOI](https://doi.org/10.1007/s12559-009-9009-8), [PDF consultado](https://rctn.org/vs265/kanerva09-hyperdimensional.pdf).

Hay tres modos y ocho configuraciones declaradas:

- `bipolar` utiliza el código de la ruta de cada observación.
- `constant_role` utiliza siempre c₀. Permite separar el uso del régimen del cambio de rasgos introducido antes de DPFP.
- `projected_concat` concatena `[x, cᵣ / 8]`, aplica una matriz fija de 128 × 64 signos escalada por 1/8 y normaliza el resultado en L2, o devuelve cero si la proyección es cero. Usa otro dominio PCG64, `[0xB3, 1, seed]`. No se declara una inversa para este control.

Las configuraciones contrastan régimen frente a calendario, rol constante, concatenación y B1 sin ligadura. La referencia lineal usa el mismo factor sin áreas. La combinación B2+B3 se contrasta con B2 sin B3, conservando los mapas. Se igualan las dimensiones de las claves y los coeficientes de los backends pertinentes. La concatenación tiene una tabla adicional y otro coste, que deben informarse.

Una ligadura constante en una memoria lineal completa puede ser solo un cambio ortogonal de coordenadas. Antes de DPFP, incluso esa ligadura constante modifica los pares de signos que generan rasgos. Por eso no basta con compararla con B1 sin factor. La inversa exacta se refiere al producto anterior a DPFP. No recupera información perdida en la proyección, en una superposición de asociaciones o en las escrituras posteriores.

Existe además un contraejemplo sencillo a una separación garantizada entre roles. Si x solo tiene una coordenada no nula, todos los productos x ⊙ cᵣ son colineales y su similitud absoluta normalizada vale uno. La prueba correspondiente conserva este caso adversarial. Tener códigos distintos no demuestra que todas las claves resultantes queden separadas.

## Disponibilidad y estados

El rol utiliza la regla de [regimes.py](../../src/mars_titan/memory/regimes.py), calculada una vez con todas las ventanas de precios del evento. El control de calendario conserva la clasificación de ausencias de esa regla. Una cohorte sin datos suficientes usa la ruta 0, identificada como sin clasificar. No se inventa un régimen.

La ruta se guarda junto a la clave y la predicción del núcleo al emitir. Cuando madura una etiqueta, se aplica el mismo operador con aquella ruta. No se recalcula usando el contexto del evento de maduración. Todas las predicciones de un evento leen el mismo estado anterior y las escrituras se publican después. El valor escrito sigue siendo etiqueta madura menos predicción del núcleo, no el error de una emisión corregida.

Los pesos, el cursor y las colas conservan su productor anterior. El estado exportado de B3 incluye además la identidad del factor y los hashes de sus tablas fijas. La recuperación rechaza otra fuente, semilla, configuración, contador, cursor o huella. La sesión incorpora los módulos del operador y de su regla temporal a su identidad de código.

La implementación actual admite régimen y calendario. No integra aún roles por mercado, modalidad o sector histórico. El `SectorMap` inspeccionado declara `point_in_time=False`, por lo que sus clasificaciones actuales no se utilizan como sectores históricos. Esto no establece que no existan otras fuentes adecuadas.

## Hipótesis y límites

La hipótesis es que una clave sensible al régimen reduce interferencias entre asociaciones de contextos diferentes. Puede perder transferencia útil entre regímenes, generar fragmentación o empeorar el error. El control de calendario permite detectar un beneficio explicado por el reparto, mientras que el rol constante ayuda a detectar un beneficio debido al cambio de proyección.

No hay nuevos parámetros compartidos, optimizador ni pasos adicionales de Titans. El operador trabaja en CPU FP64, como la corrección que amplía. Las claves conservan gradientes respecto a su entrada cuando esta los tiene. Las escrituras se desacoplan del grafo mediante el backend existente. No se ha implementado un kernel nativo adicional, porque falta evidencia de un cuello de botella que lo justifique.

La verificación técnica usa fixtures identificados. No constituye entrenamiento ni demuestra mejora predictiva, reproducción del cerebro, ausencia de interferencias o superioridad sobre otras arquitecturas. Las comparaciones reales, la interoperabilidad CUDA y los roles pendientes deben resolverse antes de dar por completado el estudio B3.

## Comprobaciones y coste del 26 de octubre de 2026

La revisión corregida aprobó 192 pruebas técnicas y, en otra ejecución, las 17 guardas del bloqueo. Los cuatro casos CUDA se deseleccionaron porque la GPU seguía ocupada. Se detectaron siete mutaciones dirigidas con una referencia verde antes y después de cada una. Incluyen omitir el rol al escribir una etiqueta, aceptar otra identidad al recuperar y reutilizar un flujo aleatorio entre semillas. Las fases no intentaron ejecutar pasos de optimizador.

El módulo nuevo cubre 101/101 sentencias y 18/18 ramas con Coverage.py 7.16.1. Radon 6.0.1 da complejidad máxima 17 y CRAP máximo 17. La cobertura de las condiciones booleanas no equivale a todas sus combinaciones. El informe de integración conserva por separado las partes no cubiertas de los módulos existentes. Quince casos de B6/B1/B2 sin B3 mantienen lecturas, estados y recuperación bit a bit frente a la revisión `3e89cc5df`.

El discriminador del generador se sitúa antes de la semilla en los dos dominios. Una prueba demostró que la forma anterior hacía coincidir la proyección de la semilla 0 y los códigos de la semilla 1, por el tratamiento de ceros finales de NumPy. La regresión detecta esa coincidencia. Las medidas siguientes corresponden a la forma corregida y a fuentes identificadas por sus hashes.

Dos pasadas midieron ocho configuraciones con 64, 256 y 1.024 filas, FP64 CPU, dos hilos, un hilo entre operaciones, cinco calentamientos y cuarenta repeticiones. La tabla recoge el ciclo de construcción de claves, lectura y escritura con 256 filas. Las rutas ya están calculadas y proceden del fixture. No incluye el predictor, el cálculo del régimen ni el almacenamiento.

| Configuración | Mediana, pasada 1 (ms) | Mediana, pasada 2 (ms) | Estado numérico (bytes) |
| --- | ---: | ---: | ---: |
| B1 sin ligadura | 8,840 | 7,437 | 6.144 |
| B1 con rol de régimen | 11,958 | 13,417 | 8.704 |
| B1 con rol constante | 9,137 | 8,421 | 8.704 |
| B1 con concatenación proyectada | 13,397 | 7,891 | 74.240 |
| B2 sin ligadura | 14,269 | 17,416 | 530.432 |
| B2 con rol de régimen | 16,278 | 15,225 | 532.992 |

Construir las claves de régimen de 256 filas costó 0,216 y 0,184 ms de mediana. B1 sin ligadura costó 0,080 y 0,074 ms. El factor añade 2.560 bytes de códigos fijos. La proyección del control añade otros 65.536 bytes. Son bytes numéricos residentes, sin cabeceras de objetos ni temporales.

La tabla cubre el corrector, sus índices y las tablas fijas. Excluye la cola pendiente de la sesión. Frente a B1 sin enrutar, B3 guarda además una ruta `int64` de 8 bytes por decisión pendiente. Los costes de contenedor de esa cola no se han medido aquí.

Los procesos completos duraron 24,68 y 24,73 segundos. El pico externo fue de 749.968 y 749.608 KiB, respectivamente. Abarca todos los casos y el entorno del proceso, no una diferencia de RAM entre variantes. Las pruebas y las mutaciones habían terminado antes de estas medidas. Otras aplicaciones siguieron activas y los tiempos variaron entre pasadas. No se atribuye una aceleración a B3 ni se infiere una latencia del predictor completo de esta tabla. Los percentiles y las muestras individuales quedan en los recibos.

La [evidencia técnica](../../reports/engineering/role-bound-keys-20261026/verification.json), las [mutaciones](../../reports/engineering/role-bound-keys-20261026/mutations.json), la [cobertura y CRAP](../../reports/engineering/role-bound-keys-20261026/quality.json), la [primera medida](../../reports/engineering/role-bound-keys-20261026/benchmark-1.json), la [segunda medida](../../reports/engineering/role-bound-keys-20261026/benchmark-2.json) y la [fuente consultada](../../reports/engineering/role-bound-keys-20261026/source-audit.json) conservan su procedencia. El [esquema](../diagrams/variants/b3-role-bound-keys.png) tiene nueve comprobaciones de composición, cuatro tamaños de navegador y revisión visual de la imagen exportada y de la captura oscura de mayor tamaño. El HTML local conserva los controles del visor en inglés.

Para repetir la medida en el entorno del proyecto:

```bash
PYTHONPATH=src:. uv run --no-sync python benchmarks/role_binding_cost.py --output build/role-binding-cost.json
```

El destino debe ser nuevo. Las comprobaciones de recuperación necesitan además el enlace nativo episódico. No hay experimentos predictivos ejecutados ni una diferencia de MAE que informar para B3.
