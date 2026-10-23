# Auditar ventanas fuente para la política de mecanismos

El protocolo de ajuste admite ahora `native_windows` como modalidad explícita. Una `MechanismFitWindow` selecciona generaciones de una sesión nativa y fija hasta dónde se pueden leer sus resultados. La modalidad anterior, `closed_phases`, sigue siendo el valor por defecto y conserva su identidad v1. La huella del caso de regresión anterior sigue siendo `738cf57a55e5550754c330e3dd45a0f7a5854db142aea9c5dfe966dca017bca3`.

El [contrato](../../../docs/engineering/mechanism-policy-fitting.md) separa dos recorridos. La auditoría independiente valida procedencia y cuenta filas maduras, pendientes, excluidas y censuradas. Los lectores protegidos prepararían ganancias para el ajuste cuando se permita. Estos últimos, la normalización, el solver y el recibo de ajuste por ventanas todavía no tienen una ejecución positiva. Este incremento no acredita un selector aprendido ni una mejora predictiva.

Cada decisión debe estar dentro del tramo declarado. La recepción real en el prefijo también debe preceder a su final exclusivo. Una etiqueta disponible antes pero recibida después no entra en esa vista. Añadir resultados o cerrar la sesión más tarde conserva el mismo prefijo histórico. Las peticiones comparten límites acumulados de bytes y cohortes y rechazan repetir mercado y fecha, incluso entre fuentes distintas. El presupuesto de bytes no es una cota del pico de RAM.

## Comprobaciones

La [regresión seleccionada](verification.json) aprobó 180 casos en 284,34 segundos, con 24 marcados CUDA, sin fallos, errores ni omisiones. Incluye 23 casos nuevos, uno CUDA. No es toda la suite del proyecto. Comprueba límites temporales y de bytes, etiquetas tardías, exclusiones, censuras, recuperación, RNG, identidad del protocolo, fuentes cambiadas y rechazo del ajuste antes de inspeccionar fuentes bajo la protección vigente.

Las funciones compartidas de transporte se comprueban con métricas dadas. Conservan ganancias negativas, ausencias, orden temporal y pesos por mercado. No calculan esas ganancias desde fuentes ni ajustan parámetros. Los pesos se reparten entre cohortes con alguna etiqueta madura y cada objetivo aplica después su máscara y normalización.

Un caso adversarial detectó que un callback sustituido podía ejecutarse antes de comprobar la integridad de la sesión. Se añadió la comprobación previa. El caso falló antes del cambio y pasó después, sin invocar el callback alterado. La primera regresión también detectó que faltaba registrar la nueva guarda en el inventario. Se añadió su prueba al inventario, sin introducir una exención.

La [mutación dirigida](mutations.json) detectó los 17 mutantes seleccionados. Las referencias inicial y restaurada aprobaron 48 casos CPU cada una. Los mutantes alteran presupuestos acumulados, cohortes repetidas, límites temporales, padre, prefijo, cabeza, censuras, identidad, guardias, signos, ausencias, pesos, orden y comprobación de callbacks. Los dos mutantes de guardias fallan por alcanzar una excepción posterior en vez de rechazar antes de inspeccionar el protocolo. No se levantó la protección ni se ejecutó un ajuste para comprobarlos. No es una puntuación sobre todos los mutantes posibles.

La [cobertura y CRAP](quality.json) usan coverage.py 7.16.1, Radon 6.0.1 y `scripts/summarize_python_quality.py`. Se midieron 175/288 sentencias y 39/60 ramas de tres módulos. El nuevo auditor cubrió 76/76 sentencias y 14/14 ramas. Su iterador tiene CRAP 11 y su resumen, 9. CRAP usa cobertura de sentencias y publica las ramas aparte. Estos valores no demuestran haber probado todas las combinaciones de operandos.

El ajuste protegido conserva cobertura limitada al rechazo. `fit_policy` tiene CRAP 442,83, `_cohorts` 144,01 y `_window_cohorts` 9,56. La ejecución numérica positiva pendiente sigue siendo un límite de la verificación. No se presenta la cobertura del auditor como cobertura del aprendizaje completo.

## Coste observado

El [protocolo](cost-protocol.json) se guardó antes de medir. Las medidas registran el HEAD padre `b449659b3fd1fab21da93aac669545d53df65bed` y las huellas SHA-256 de las fuentes modificadas que se ejecutaron. Esas huellas identifican el código medido, que todavía no tenía su commit al cronometrarlo. Se completaron dos pasadas con 4, 64 y 256 filas lógicas y una, dos o cuatro colecciones. Cada condición compara una petición conjunta con peticiones separadas del mismo prefijo. Se comprobó igualdad exacta de paquetes, resultados, metadatos y recuentos antes de cronometrar. Cada modo tuvo tres calentamientos y veinte repeticiones, con orden alterno. Son 36 condiciones y 720 muestras temporizadas.

El productor usa `cuda:0`, FP64, una RTX 4070 Laptop de 8.188 MiB, Torch 2.14.0+cu130 y CUDA 13.0, con driver 595.91.07 y dos hilos CPU. TF32 y autocast están desactivados. La auditoría trabaja con JSON y archivos nativos en CPU. No se ha trasladado ese trabajo a GPU sin evidencia de que compense.

Las filas replican el mismo fixture de precios, con noticias, fundamentales y macro ausentes y máscaras explícitas. Las etiquetas son valores manuales identificados. Las 256 filas se procesan en bloques de 64 y no son 256 historias bursátiles independientes ni el corpus FinMultiTime. El productor permite la actualización asociativa interna de memoria durante inferencia. No hay pasos de optimizador sobre parámetros compartidos ni entrenamiento del controlador.

Las medianas de auditar cuatro colecciones fueron las siguientes, en milisegundos:

| Filas | Conjunta, primera | Separadas, primera | Conjunta, segunda | Separadas, segunda |
| --- | ---: | ---: | ---: | ---: |
| 4 | 33,41 | 82,07 | 30,07 | 70,60 |
| 64 | 105,38 | 198,59 | 104,90 | 196,45 |
| 256 | 467,53 | 673,83 | 333,31 | 532,13 |

Con 256 filas y cuatro colecciones, los p95 conjunta/separadas fueron 857,77/1.381,96 ms y 402,60/708,35 ms. Los p99 fueron 952,31/1.384,61 ms y 471,72/716,21 ms. Veinte muestras dan colas descriptivas y hay variación entre pasadas. Con una sola colección ambos modos realizan la misma petición y no hay una ventaja consistente. El [resumen de coste](cost.json) conserva todos los tamaños, caudales y extremos.

Los bytes lógicos de paquetes y registros pasan de 2.671.661 en peticiones separadas a 1.503.649 en la conjunta, para 256 filas y cuatro colecciones. No incluyen los bloques de trazas de integridad ni equivalen a lecturas físicas. El pico adicional Python medido aparte con `tracemalloc` fue casi igual: 7.168.616/7.171.041 bytes en la primera pasada y 7.170.465/7.170.339 en la segunda. No se concluye una reducción de RAM en ese caso. Durante la transición entre ventanas pueden coexistir referencias a la anterior y a la siguiente.

El pico asignado por Torch durante las consultas fue de 67.395.072 bytes, igual al residente observado. No es la VRAM total ni el pico de creación de los fixtures. El RSS interno fue de 1.554.464.768 bytes en ambas pasadas. Los picos externos fueron de 1.614.840 y 1.615.152 KiB. Los [tiempos de proceso](process-cost.json) fueron 114,14 y 117,77 segundos internos y 118,13 y 122,43 externos.

La latencia incluye validaciones del consumidor, cabeza, lectura y reconstrucción. Excluye crear las sesiones y emitir sus predicciones. El tiempo de proceso incluye esa preparación y las medidas separadas de asignaciones. No hubo otras pruebas, mutaciones ni consultas propias en paralelo con el cronometraje. No se detuvieron otras aplicaciones y había carga ajena visible en la GPU antes de medir. No se midieron energía ni contadores de caché, transferencias físicas u ocupación.

## Reproducción y alcance pendiente

Con el entorno uv y la extensión episódica nativa del proyecto configurados, se puede reproducir una pasada con una salida y un directorio externo nuevos:

```bash
evidence_dir="$(mktemp -d)"
mkdir "$evidence_dir/fixtures"
uv run --no-sync python reports/engineering/mechanism-fit-window-20261021/benchmark.py --output "$evidence_dir/cost.json" --storage-root "$evidence_dir/fixtures"
```

El script exige CUDA para producir los fixtures y rechaza una salida existente. [external-artifacts.json](external-artifacts.json) identifica por tamaño y SHA-256 los XML, registros, cobertura, muestras, fuentes medidas e inventario externo de los archivos nativos. Conserva también el intento inicial con importación incorrecta y el caso RED de la API ausente. Esos intentos no se suman a la regresión final.

La selección de mecanismos aprendida desde ventanas, su validación temporal posterior y el contraste con reglas simples siguen pendientes. No se han abierto el corpus para una campaña nueva ni el test final. El resultado de este incremento es una auditoría temporal comprobada y una ruta de integración preparada. No demuestra superioridad predictiva, novedad frente a otros modelos ni una reproducción de memoria cerebral.
