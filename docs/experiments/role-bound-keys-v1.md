# Claves ligadas a roles observables

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

Hay tres modos y doce configuraciones declaradas:

- `bipolar` utiliza el código de la ruta de cada observación.
- `constant_role` utiliza siempre c₀. Permite separar el uso del régimen del cambio de rasgos introducido antes de DPFP.
- `projected_concat` concatena `[x, cᵣ / 8]`, aplica una matriz fija de 128 × 64 signos escalada por 1/8 y normaliza el resultado en L2, o devuelve cero si la proyección es cero. Usa otro dominio PCG64, `[0xB3, 1, seed]`. No se declara una inversa para este control.

Las configuraciones contrastan régimen frente a calendario, rol constante, concatenación y B1 sin ligadura. Las fuentes por fila se contrastan con el rol constante, y el sector también con su reparto permutado. La referencia lineal usa el mismo factor sin áreas. La combinación B2+B3 se contrasta con B2 sin B3, conservando los mapas. Se igualan las dimensiones de las claves y los coeficientes de los backends pertinentes. La concatenación tiene una tabla adicional y otro coste, que deben informarse.

Una ligadura constante en una memoria lineal completa puede ser solo un cambio ortogonal de coordenadas. Antes de DPFP, incluso esa ligadura constante modifica los pares de signos que generan rasgos. Por eso no basta con compararla con B1 sin factor. La inversa exacta se refiere al producto anterior a DPFP. No recupera información perdida en la proyección, en una superposición de asociaciones o en las escrituras posteriores.

Existe además un contraejemplo sencillo a una separación garantizada entre roles. Si x solo tiene una coordenada no nula, todos los productos x ⊙ cᵣ son colineales y su similitud absoluta normalizada vale uno. La prueba correspondiente conserva este caso adversarial. Tener códigos distintos no demuestra que todas las claves resultantes queden separadas.

## Disponibilidad y estados

El rol utiliza la regla de [regimes.py](../../src/mars_titan/memory/regimes.py), calculada una vez con todas las ventanas de precios del evento. El control de calendario conserva la clasificación de ausencias de esa regla. Una cohorte sin datos suficientes usa la ruta 0, identificada como sin clasificar. No se inventa un régimen.

La ruta se guarda junto a la clave y la predicción del núcleo al emitir. Cuando madura una etiqueta, se aplica el mismo operador con aquella ruta. No se recalcula usando el contexto del evento de maduración. Todas las predicciones de un evento leen el mismo estado anterior y las escrituras se publican después. El valor escrito sigue siendo etiqueta madura menos predicción del núcleo, no el error de una emisión corregida.

Los pesos, el cursor y las colas conservan su productor anterior. El estado exportado de B3 incluye además la identidad del factor y los hashes de sus tablas fijas. La recuperación rechaza otra fuente, semilla, configuración, contador, cursor o huella. La sesión incorpora los módulos del operador y de su regla temporal a su identidad de código.

## Fuentes por fila

Desde el 30 de octubre de 2026 B3 admite cuatro fuentes más, que dan una ruta propia a cada fila del evento. Están en [role_sources.py](../../src/mars_titan/memory/role_sources.py) y ninguna ajusta parámetros con datos. Siguen la decisión 9 del estudio, que pide primero modalidad y mercado y después grupos por correlación de la ventana pasada ([C4](pit-region-gate-v1.md) la explica con más detalle).

- `market` usa la ruta 1 para EE. UU. y la 2 para China, según el prefijo del flujo. En una población de un solo mercado, como la de la criba, todas las filas comparten ruta y el rol equivale al rol constante con otro código. Solo informa en una comparación con los dos mercados.
- `modality` usa la ruta 1 + noticias + 2·fundamentales con los bits de presencia de la fila. Precios y gráficos son obligatorios y el macro está presente en todas las filas de la edición, así que no distinguen filas. Con la política estricta todas las filas van a la ruta 4.
- `sector` es un sector estadístico en fecha. Los activos de cada mercado se agrupan por la correlación de sus rendimientos diarios en la ventana de 64 sesiones que termina en la decisión, con la regla común de C3 y C4 ([correlation_regions.py](../../src/mars_titan/data/correlation_regions.py)) cortada en cuatro grupos. Solo cuentan los rendimientos entre dos sesiones presentes. Los activos con menos de 42 rendimientos, el mínimo del régimen, o con la serie constante van a la ruta 0.
- `permuted_sector` es el control del sector. Reparte las mismas rutas y tamaños al azar entre los activos con historia de cada mercado. El generador depende solo de la semilla y del mercado, así que con los mismos activos cada uno recibe siempre la ruta del mismo compañero. Conserva la ocupación de cada evento y retira la estructura de correlación.

Los grupos del sector se numeran por el menor activo que contienen. Una partición estable conserva sus rutas, pero un cambio de miembros puede renumerarlas, y entonces el mismo código pasa a designar otro grupo. Por eso el informe del recorrido cuenta, con estas fuentes, los cambios de ruta de cada activo entre dos decisiones suyas, igual que cuenta los del mercado con el régimen. Además, con cuatro grupos fijos una cohorte con poca estructura se parte igualmente en cuatro por diferencias pequeñas de correlación. El contraste con el reparto permutado debe mostrar si la partición aporta algo más que su ocupación.

El sector de las fichas actuales (`SectorMap`) no se usa como fuente. Declara `point_in_time=False` y tiene once sectores en EE. UU. y 31 en China, más un grupo desconocido en cada mercado, que no caben en los cuatro compartimentos clasificados de B3. C4 y H4 ya lo miden como control de fuga, el único uso que admite la decisión 9. Esto no establece que no existan otras fuentes sectoriales históricas adecuadas.

Las fuentes por fila comparten el recorrido de las anteriores. La ruta de cada fila se calcula una vez con todo el evento, se guarda al emitir y se reutiliza al madurar la etiqueta. La ventana y `FinancialSession` pasan además los bits de presencia de cada fila. La sesión incluye en su identidad de código el módulo de las fuentes y el de las regiones por correlación cuando B3 está activo. También incluye ahora la regla de régimen con cualquier clave enrutada de B6, que antes solo entraba con B3.

### Coste de las fuentes por fila

La [medida del 30 de octubre](../../reports/engineering/role-sources-20261030/cost-cpu.json), en `0849e9b65`, calcula la ruta del evento completo con ventanas sintéticas `[filas, 64, 6]` en FP32, los cinco bits de presencia y flujos del mercado US. La semilla de las ventanas es `20261030 + filas`. Usa tres calentamientos y veinte repeticiones, alternando las seis fuentes con orden rotatorio. No modifica parámetros ni mide error predictivo. Cada celda da la mediana en milisegundos y, entre paréntesis, el percentil 95 descriptivo:

| Fuente | 64 filas | 256 filas | 1.024 filas |
| --- | ---: | ---: | ---: |
| Régimen | 0,268 (0,293) | 1,183 (1,641) | 2,688 (4,932) |
| Calendario | 0,212 (0,270) | 0,988 (1,374) | 2,523 (4,689) |
| Mercado | 0,038 (0,039) | 0,175 (0,260) | 0,417 (0,801) |
| Modalidad | 0,035 (0,041) | 0,163 (0,247) | 0,417 (0,905) |
| Sector estadístico | 0,647 (0,697) | 7,863 (11,127) | 62,358 (109,631) |
| Sector permutado | 0,698 (0,755) | 8,250 (11,927) | 65,911 (115,370) |

La agrupación por correlación domina este coste con 1.024 filas. La tabla mide el cálculo de rutas, que ocurre una vez por evento, y no el ciclo posterior de claves, lectura y escritura ni el predictor completo. Tampoco informa de una ventaja del rol de mercado: con todos los flujos en US es constante. Las ventanas tienen todas sus sesiones presentes y no representan todas las ausencias del corpus.

El tramo medido dura 4,382 s en un Ryzen 9 8945HS, con NumPy 2.5.3, SciPy 1.18.1, Python 3.12.14 y dos hilos declarados. La carga media del equipo estaba entre 6,7 y 8,2. No se midieron el tiempo total del proceso, los picos de RAM o VRAM ni la energía. Veinte repeticiones no acreditan una cola p95 estable. El recibo conserva cada muestra, las ocupaciones y las huellas de las fuentes, sin atribuir una aceleración a B3.

```bash
PYTHONPATH=src:. uv run --no-sync python -m benchmarks.role_sources_cost \
  --sizes 64 256 1024 --warmup 3 --repetitions 20 --output build/role-sources-cost.json
```

El destino debe ser nuevo.

## Hipótesis y límites

La hipótesis es que una clave sensible al régimen reduce interferencias entre asociaciones de contextos diferentes. Puede perder transferencia útil entre regímenes, generar fragmentación o empeorar el error. El control de calendario permite detectar un beneficio explicado por el reparto, mientras que el rol constante ayuda a detectar un beneficio debido al cambio de proyección.

No hay nuevos parámetros compartidos, optimizador ni pasos adicionales de Titans. El operador trabaja en CPU FP64, como la corrección que amplía. Las claves conservan gradientes respecto a su entrada cuando esta los tiene. Las escrituras se desacoplan del grafo mediante el backend existente. No se ha implementado un kernel nativo adicional, porque falta evidencia de un cuello de botella que lo justifique.

La verificación técnica usa fixtures identificados. No constituye entrenamiento ni demuestra mejora predictiva, reproducción del cerebro, ausencia de interferencias o superioridad sobre otras arquitecturas. Las comparaciones reales y la interoperabilidad CUDA deben resolverse antes de dar por completado el estudio B3.

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

## Comprobaciones del 30 de octubre de 2026

La [verificación de las fuentes por fila y E4](../../reports/engineering/role-sources-20261030/verification.json) conserva 525 pruebas CPU aprobadas sin omisiones sobre la integración con `develop`. Comprueban rutas por fila, ausencias, orden de activos, identidad, gradientes, recuperación, paridad de las variantes desactivadas y consumidores compartidos. Una pasada anterior independiente también aprobó 525 casos. Son selecciones solapadas y no se suman.

Las 48 mutaciones de B3 se repitieron después de incorporar E4. En esa repetición, tres casos de coste quedaron inconclusos porque la copia aislada no podía consultar el commit. Al repetir solo esos tres con metadatos Git reales de solo lectura, todos dieron referencias verdes antes y después y el fallo esperado al mutar. El total final sigue siendo 48 identificadores distintos, más 28 propios de E4, sin intentos de optimizador. Esta muestra dirigida no es una puntuación de mutación de todo el proyecto.

La cobertura de doce módulos suma 2.345/2.719 sentencias y 781/1.048 ramas. Se midieron las 40 funciones cambiadas, con CRAP máximo 145,94 en `FinancialSession.__init__`. La selección no recorre todas las rutas previas de esa clase. Ruff, formato y el comprobador del repositorio pasan. La regresión histórica de importadores registró 2.041 aprobados y 42 omitidos en training, y 7.128 aprobados, 206 omitidos y un fallo de aislamiento del fixture de determinismo en el resto. Ese fallo se corrigió en #819 y su módulo pasa en la selección integrada. No se presenta la regresión histórica completa como repetida ni libre de omisiones.

Las comprobaciones CUDA siguen en #816. Los costes de rutas publicados no miden el predictor completo ni el error sobre datos reales. El aprendizaje y el test final permanecen bloqueados y no hay mejora predictiva medida.

Para repetir la medida en el entorno del proyecto:

```bash
PYTHONPATH=src:. uv run --no-sync python benchmarks/role_binding_cost.py --output build/role-binding-cost.json
```

El destino debe ser nuevo. Las comprobaciones de recuperación necesitan además el enlace nativo episódico. No hay experimentos predictivos ejecutados ni una diferencia de MAE que informar para B3.
