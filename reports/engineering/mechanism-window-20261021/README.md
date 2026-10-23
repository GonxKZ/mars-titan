# Leer varias colecciones desde un prefijo nativo

`FinancialSession.mechanism_collections_at` reconstruye varias colecciones de alternativas desde las mismas generaciones confirmadas. Lee cada registro nativo una vez, dirige cada resultado por flujo y fecha y devuelve una procedencia común. Sirve para preparar ventanas temporales del controlador dentro de una sesión más larga. No ajusta parámetros.

El código está en `abd288ee0d29087e2c621f6d82cfb6b25a314d20`. El [contrato](../../../docs/engineering/mechanism-collection-asof.md) describe generaciones, límites y salida. La API individual usa el mismo motor de recepción y conserva su esquema. La comparación con la fuente anterior comprueba también su huella completa, no solo sus resultados financieros.

## Comprobaciones ejecutadas

La [regresión seleccionada](verification.json) pasó 222 casos en 464,89 segundos, con 33 marcados CUDA, sin fallos, errores, avisos ni omisiones. Incluye 24 casos nuevos de la ventana, dos CUDA. No es toda la suite del proyecto. Se ejecutó con cobertura de ramas sobre el directorio de producción y se filtraron después los dos módulos afectados.

Los casos nuevos comprueban resultados recibidos tarde, corte exclusivo, selecciones no consecutivas, identidad por fecha, censuras sin etiqueta inventada, límites exactos de bytes, recuperación y cambio de cabeza durante el recorrido. La selección de generaciones no consulta los recibos actuales ni carga paquetes no elegidos. Una sonda de lectura nativa, que delega en la función real, comprueba que cada registro se abre una sola vez. No sustituye los métodos protegidos de la sesión.

La [mutación dirigida](mutations.json) detectó 15 de 15 mutantes. Las referencias inicial y restaurada aprobaron 43 casos CPU cada una, sin errores ni omisiones. Los cambios introducidos mezclan fechas, leen futuro, alteran el corte exclusivo, ignoran huellas o presupuestos, omiten censuras y procedencia, cargan paquetes no seleccionados, duplican lecturas o aceptan una cabeza cambiada. No es una puntuación sobre todos los mutantes posibles.

La [calidad](quality.json) registra 567/688 sentencias y 184/270 ramas en 43 funciones. Se usaron coverage.py 7.16.1, Radon 6.0.1 y el cálculo común `scripts/summarize_python_quality.py`. CRAP utiliza cobertura de sentencias y publica las ramas aparte. El módulo del lector cubre 99/99 sentencias y 20/20 ramas, con CRAP máximo 12. El nuevo método de sesión cubre 10/10 y 4/4, con CRAP 4. La selección no recorre todas las rutas anteriores de `FinancialSession`. Su método `_associative_state` conserva 17 sentencias sin recorrer y CRAP 272. Una condición cubierta no implica haber comprobado cada combinación de sus operandos.

El primer caso RED falló por la API ausente. La primera pasada posterior tuvo 45 aprobados y un fallo del test de instrumentación, que intentaba sustituir un método protegido. Se trasladó la sonda al límite de lectura nativa, sin retirar la protección. La pasada de ambos lectores aprobó sus 51 casos antes de la regresión final. Esas tandas no se suman a los 222 casos finales.

## Coste observado

El [protocolo](cost-protocol.json) se registró antes de cada pasada. Compara la fuente individual anterior de `ba6395cedaa7117bffb9cef395a3ea695032e820`, la API individual actual y la ventana. La fuente anterior se carga en el mismo entorno actual y con las mismas comprobaciones de sesión. No se reproduce todo el entorno histórico de ese commit. Se midieron 4, 64 y 256 activos lógicos, con una, dos y cuatro colecciones. Cada modo tuvo tres calentamientos y veinte repeticiones, con orden rotado por repetición. Se conservaron ambas pasadas completas, 54 condiciones y 1.080 muestras temporizadas. El [resumen](cost.json) incluye todos los tamaños, modos, colas y extremos.

El productor de los fixtures usa `cuda:0`, FP64, Torch 2.14.0+cu130, CUDA 13.0 y una RTX 4070 Laptop de 8.188 MiB, con driver 595.91.07 y dos hilos CPU. La lectura trabaja con JSON y archivos nativos en CPU. TF32 y autocast están desactivados. Las 256 filas lógicas se procesan en bloques de 64. No se amplió el límite de lote del predictor.

Los activos replican el mismo fixture de precios, con noticias, fundamentales y macro ausentes y máscaras explícitas. Las etiquetas son valores manuales identificados. No son 256 historias bursátiles independientes ni el corpus FinMultiTime. La fase definitiva contiene las cuatro decisiones y sus disponibilidades antes de `decision_end`, con cierre posterior. Los tres modos reciben exactamente las mismas filas y resultados en cada condición. No se han ejecutado pasos de optimizador sobre parámetros compartidos. Las actualizaciones asociativas de memoria rápida pertenecen a la inferencia del fixture.

Durante las medidas no hubo otras pruebas, mutaciones, perfiles ni consultas propias en paralelo. No se detuvieron otras aplicaciones. La GPU tenía carga ajena visible antes de las pasadas. No se atribuye exclusividad al equipo. Los tiempos incluyen las comprobaciones del consumidor y de la cabeza, la lectura, la validación y la construcción de la salida. No incluyen crear el fixture ni emitir las predicciones. El tiempo total de proceso sí incluye esa preparación y el perfil de asignaciones separado.

Estas son las medianas de la consulta de cuatro colecciones, en milisegundos:

| Activos | Individuales, primera | Conjunta, primera | Individuales, segunda | Conjunta, segunda |
| --- | ---: | ---: | ---: | ---: |
| 4 | 60,42 | 27,72 | 68,73 | 32,04 |
| 64 | 122,10 | 64,78 | 248,88 | 142,47 |
| 256 | 449,25 | 245,67 | 440,26 | 239,10 |

Con 256 activos y cuatro colecciones, el p95 individual/conjunto fue 530,42/446,37 ms en la primera pasada y 457,85/309,47 ms en la segunda. El p99 fue 552,37/455,05 y 496,91/355,35 ms. El caudal pasó de 9,09 a 15,04 colecciones/s y de 9,79 a 16,73. Con veinte muestras, las colas son descriptivas. La dispersión entre pasadas, visible también con 64 activos, impide presentar un coste universal.

La referencia anterior de 256 activos y cuatro colecciones tuvo p50 de 453,84 y 444,23 ms. La API individual actual tuvo 449,25 y 440,26 ms, con igualdad exacta de sus salidas y huellas. En otras condiciones se observaron aumentos de su mediana, hasta un 7,77 % en 256 activos y dos colecciones de la segunda pasada. No se afirma ausencia de regresión temporal en todas las formas. Con una sola colección tampoco hay un beneficio consistente de la ventana. No se ha cambiado el lector científico para activarla automáticamente.

La ventana de cuatro colecciones recorre cinco registros nativos. Solicitar las cuatro vistas por separado recorre sus prefijos y repite registros, con catorce lecturas según las generaciones elegidas. Los paquetes y registros contabilizados pasan de 2.671.661 a 1.503.649 bytes en el caso de 256 activos. Esta suma no incluye los bloques de trazas ni mide bytes físicos del almacenamiento.

El pico de asignaciones Python, medido aparte de las latencias con `tracemalloc`, pasa de 6.819.609 a 7.164.008 bytes en la primera pasada y de 6.819.499 a 7.164.517 en la segunda. La ventana reduce recorridos, pero conserva varios paquetes para devolverlos juntos. No reduce toda la memoria. `tracemalloc` tampoco cubre las asignaciones nativas ni toda la RAM.

El pico asignado por Torch durante las consultas fue de 67.395.072 bytes, igual al estado residente observado, en los tres modos. No es la VRAM total ni el pico de creación de las sesiones. Los [tiempos de proceso](process-cost.json) fueron 123,07 y 148,08 segundos internos y 126,87 y 152,22 externos. Los picos RSS Python fueron 1.557.426.176 y 1.555.681.280 bytes. Los externos fueron 1.617.324 y 1.617.360 KiB. No se midieron energía, ocupación ni contadores de caché del hardware.

## Ejecuciones incompletas y reproducción

Se conservan dos intentos incompletos del script. El primero intentó emitir una decisión en `decision_end`, que es exclusivo. El segundo dejó la última disponibilidad de etiqueta en esa misma frontera, también prohibida por el ejecutor financiero. Ambos terminaron con salida 1 tras medir solo los tamaños iniciales. No se usan como pasadas completas ni se cambió el motor para aceptarlos. Una comprobación posterior verificó las cuatro colecciones y sus dieciséis etiquetas manuales antes de repetir la medida completa.

La fase definitiva usa `moment(126)` como primera decisión, `moment(132)` como final exclusivo, `moment(133)` como cierre y `moment(134)` como corte de la vista. `moment` procede del fixture y del calendario existente. No son fechas de un experimento financiero nuevo.

Con el entorno uv del proyecto y la extensión episódica nativa configurados, el benchmark se reproduce así:

```bash
evidence_dir="$(mktemp -d)"
git show ba6395cedaa7117bffb9cef395a3ea695032e820:src/mars_titan/memory/mechanism_collection_asof.py > "$evidence_dir/reference-asof.py"
uv run --no-sync python reports/engineering/mechanism-window-20261021/benchmark.py --reference "$evidence_dir/reference-asof.py" --output "$evidence_dir/cost.json"
```

El script rechaza una referencia cuyo SHA-256 no sea `7709cd573ad327c8579674704a06164d1ff365d6d9ecafc33fa035b5a69103dd`. Las muestras, XML, cobertura y registros se conservan fuera de Git. [external-artifacts.json](external-artifacts.json) identifica sus tamaños y huellas, incluidos los intentos fallidos.

El lector de ajuste sigue exigiendo sesiones completas cerradas dentro de su tramo. La integración de esta ventana, la validación positiva de normalización y solver y el aprendizaje útil del controlador siguen pendientes. La campaña completa y el test final no se han abierto. Este incremento no mide una mejora predictiva, una especialización regional ni una reproducción de memoria cerebral.
