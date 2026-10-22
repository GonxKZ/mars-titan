# Validación de la geometría regional al cargar checkpoints

22 de octubre de 2026. Código comprobado en `8e3dc4c8433bfcd9ebdb54ddce0b03b22cab60a0`, con las dependencias de la PR #708. Este arreglo es un prerrequisito de #709. No completa sus programas regionales ni el ciclo de propuestas de #711.

`RegionalFusion` comparaba las aristas entrantes con el buffer mutable de la instancia. Alterar ambos del mismo modo permitía aceptar otra topología bajo la configuración original. Además, el predictor podía copiar pesos antes del rechazo del hijo. La reproducción inicial falló en los 20 casos seleccionados. Las ampliaciones detectaron otras rutas con formas internas alteradas, NaN diferidos y metadatos de asignación contradictorios. No se ha demostrado corrupción de checkpoints históricos.

La carga deriva las aristas de la configuración y comprueba estructura, formas y finitud antes de copiar. El sellado del predictor comprueba también la geometría instalada. Un checkpoint genuino puede reparar valores booleanos alterados. No se acepta como reparación un buffer de tipo, layout o forma incompatibles. Las APIs públicas rechazan metadatos de asignación no booleanos o que contradigan el argumento explícito. La asignación regional exige un dispositivo común. La copia ordinaria conserva el dispositivo del destino y admite orígenes CPU/CUDA mezclados.

El rechazo previo se ha comprobado en estas APIs directas. No se afirma atomicidad universal ante hooks arbitrarios o contenedores externos que eludan la entrada pública. Una conversión rechazada puede haber convertido tensores, pero no obtiene un sello nuevo que legitime la geometría alterada.

## Comprobaciones

La selección final contiene 348 casos de siete archivos, con cero fallos, errores u omisiones. Incluye 154 casos nuevos de carga, estructura, metadatos y conversiones. Se ejecutaron CPU y `cuda:0`, con FP32 y FP64 donde corresponde. El único aviso corresponde al modo de depuración de sincronizaciones de PyTorch, que declara incompleta su detección de operaciones sincronizantes. El [recibo de verificación](verification.json) fija las fuentes y pruebas.

La [paridad con la referencia anterior](parity.json) cubre 16 condiciones: CPU/CUDA, FP32/FP64 y fusión desactivada, diagonal, anillo o completa. Coinciden exactamente parámetros, claves del checkpoint, configuración arquitectónica, RNG CPU y los resultados completos de dos emisiones consecutivas. Los costes comprueban además RNG CUDA y resultados completos. Cambian los hashes de fuente y los identificadores del consumidor. La compatibilidad de pesos no autoriza intercambiar runtimes históricos como si tuvieran el mismo sello.

La [mutación dirigida](mutations.json) detectó nueve de nueve sustituciones elegidas. Cada selección ejecutó un caso y falló en `call` con el nodeid, parámetros, excepción y mensaje previstos. Se verificaron los archivos importados y sus SHA. La referencia y la copia restaurada pasaron 154 casos cada una, sin errores, omisiones o timeouts. Producción permaneció intacta. Este resultado no es una puntuación global de mutación del proyecto.

La [cobertura y CRAP](quality.json) usan coverage.py 7.16.1 con ramas y Radon 6.0.1. Se cubren 622 de 771 sentencias y 238 de 362 ramas de los dos módulos completos. CRAP usa cobertura de sentencias, con `C² × (1 − cobertura)³ + C`. El payload regional cubre sus 17 sentencias y 12 ramas. La validación estructural cubre 14 de 16 sentencias y 6 de 8 ramas, con complejidad 15 y CRAP 15,4395. El informe conserva las funciones anteriores sin cubrir. No se declara cobertura completa.

El primer intento de instrumentación no encontró `coverage` en el entorno. El siguiente, con selección de módulos mediante `--source`, falló al importar PyTorch con una aserción interna en su integración NumPy. La ejecución final usa las versiones fijadas mediante `uv --with` y selección de archivos mediante `--include`. Ambos intentos se conservan y no cuentan como pruebas superadas.

## Coste observado

El [protocolo fijado antes de medir](cost-protocol.json) usa RTX 4070 Laptop, PyTorch 2.14.0+cu130, CUDA 13, FP64, H=32 y un hilo CPU. Compara cuatro topologías, lotes de 1 y 256, carga y preparación completa del consumidor congelado. Cada versión tiene 32 registros, tres calentamientos y diez muestras por registro, con dos órdenes invertidos. Son 208 llamadas de carga y 208 de preparación por versión. Las entradas y etiquetas son fixtures, sin corpus real, optimizador ni evaluación científica.

La referencia anterior es una copia con dos fuentes restauradas a `bbb2b51`. No es un checkout íntegro de ese commit. Los 32 pares coinciden en condiciones, checkpoint, parámetros, RNG y resultado completo. El [archivo de medidas](cost.json) conserva todas las condiciones y muestras. La tabla recoge el anillo con 256 filas.

| Recorrido | Orden | p50 anterior, ms | p50 corregido, ms |
| --- | --- | ---: | ---: |
| Carga del checkpoint residente | Normal | 9,6590 | 7,2518 |
| Carga del checkpoint residente | Inverso | 9,6544 | 12,4491 |
| Preparación completa | Normal | 14,4625 | 11,9771 |
| Preparación completa | Inverso | 22,0965 | 9,7354 |

Los procesos fueron separados y coexistieron con otras aplicaciones. La variación entre órdenes impide atribuir aceleración al arreglo. Los percentiles de diez muestras son descriptivos. Los tiempos de proceso fueron 33,5982 y 18,9687 segundos e incluyen importaciones, construcción, comprobaciones y medidas. No son una medida exclusiva del recorrido numérico.

En esta condición, el pico adicional del asignador Torch durante la preparación fue de 25.826.816 bytes en ambas versiones. Durante la carga pasó de 1.024 a 2.560 bytes. Los parámetros residentes conservaron 283.144 bytes. Los máximos RSS de proceso declarados fueron 1.412.624.384 y 1.446.473.728 bytes. Incluyen bibliotecas y otras asignaciones del proceso. No se midieron energía ni transferencias físicas. Se mantienen las primitivas CUDA existentes, sin nuevos kernels ni relajación de precisión.

El [guion reproducible](cost.py) conserva las funciones ejecutadas. Solo cambia el docstring inicial y el orden de tres imports que los fixtures ya habían cargado. El protocolo es idéntico al original. Copiar ambos fuera del repositorio evita escribir resultados voluminosos en él. Para repetir la versión actual, desde la raíz y con el entorno CUDA local y su guarda disponibles:

```bash
regional_bench_dir="$(mktemp -d)"
cp reports/engineering/regional-geometry-20261022/cost.py reports/engineering/regional-geometry-20261022/cost-protocol.json "$regional_bench_dir/"
timeout --kill-after=5s 180s uv run --no-sync python "$regional_bench_dir/cost.py" --repository "$PWD" --label after
```

El guion exige CUDA y un directorio de salida nuevo. Cada proceso comprueba una cuota interna de 170 segundos y el comando aplica un límite externo de 180 segundos. La repetición anterior exige además la copia y su manifiesto fijados, conservados con los [artefactos externos](external-artifacts.json). No se presenta esa copia como reconstruible a partir de un único commit.

No se ha entrenado un modelo ni medido una mejora predictiva en este incremento. La campaña completa y el test final continúan cerrados. Siguen pendientes el ejecutor regional con tres ritmos y presupuesto conjunto, la política de arranque dentro del soporte y el ciclo exterior de propuestas contrastables.
