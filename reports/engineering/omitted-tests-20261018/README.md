# Comprobación de las pruebas antes omitidas

Las 612 omisiones del [inventario del 17 de octubre](../validation-20261017/README.md) se han ejecutado y resuelto. Pasan 147 casos en `cuda:0` y 465 en CPU. También pasan los 28 casos añadidos y 136 regresiones adicionales. El [recibo](verification.json) conserva cada identificador, resultado, dispositivo, duración por fase y huella del registro. Las ejecuciones abarcan el 17 y el 18 de octubre de 2026, en horario de Madrid.

Se trata de pruebas locales con datos sintéticos y fixtures. Incluyen pasos reales de optimizador y ajustes HMM dentro de esa selección técnica. El corpus y los estados de las campañas quedaron fuera del espacio de archivos accesible por las pruebas, la red estuvo aislada y la protección global de aprendizaje conservó su huella. No se ejecutaron campañas reales, pilotos científicos ni el test final de 2024. Tampoco se ha repetido íntegramente la batería anterior de 12.010 casos.

## Fallos corregidos

`AnchoredAdamW` interpretaba el primer elemento de los argumentos del gancho como una clausura. PyTorch incluye ahí el propio optimizador. El gancho ahora separa ese argumento y rechaza clausuras, argumentos repetidos y opciones desconocidas antes de modificar pesos. Las pruebas CPU y CUDA comprueban el paso completo, la equivalencia exacta con AdamW cuando el ancla o el decaimiento son cero, la recuperación de anclas y momentos y la prioridad del bloqueo global.

Los ejecutores tabulares rechazaban una partición ausente al escribir las predicciones, después de haber ajustado el modelo. Ahora comprueban la compatibilidad entre manifiesto y retención antes del ajuste y de crear la salida. Los fixtures de matriz compartida utilizan vistas temporales con las particiones que declaran. La comparación XGBoost conserva igualdad de checkpoints y predicciones entre matrices propias y compartidas.

Se corrigieron marcas CUDA ausentes en pruebas que preparan matrices externas, padres Ridge o referencias neuronales. Sus cuerpos no cambiaron en los 17 archivos que solo necesitaban esas marcas. Los casos CPU independientes permanecen seleccionables por separado. La referencia CPU de selección XGBoost utiliza la validación completa, mientras que las pruebas CUDA comprueban la caché incremental. Los avisos por dispositivos incompatibles son errores en esa referencia.

Los callbacks de interrupción conservan ahora el cursor de trazas. El doble de búsqueda tabular separa el directorio de caché de la identidad serializada. Las comprobaciones de transporte FP32 vuelven a comparar exactamente los valores ampliados a FP64, conforme al transporte incorporado en `6541d94a`. No se modificó su precisión ni se ampliaron tolerancias.

La prueba de publicación usaba pesos iniciales sin historial de ajuste y el control de presupuesto los rechazaba correctamente. Su recorrido de publicación utiliza ahora cuatro ajustes reales de dos épocas, con dos activos sintéticos y dos ventanas. Un caso adicional exige que los agregados sin historial sigan siendo rechazados. La publicación, la regeneración y las comparaciones de tablas conservan sus aserciones originales.

## Memoria y diagnósticos

El fixture de campaña Titans retenía una copia de cada gradiente aunque sus comprobaciones solo consultaban llamadas y recibos. El registrador permite conservar únicamente esas llamadas. Las pruebas que comparan gradientes mantienen las copias completas por defecto. El caso de ampliación arquitectónica incluye además el campo `traces` de su petición.

| Medida del proceso de prueba | Antes | Después |
| --- | ---: | ---: |
| Pico de RAM residente | 5,433 GiB | 0,907 GiB |
| Tiempo de proceso | 516,61 s | 461,91 s |
| Resultado | Fallo al entrar en el caso, después de preparar la campaña | Caso completo superado |

La preparación conserva los mismos 14 recorridos con optimizador registrador y 12 traslados. La ejecución posterior completa además el caso que antes fallaba. Estas medidas no estiman una aceleración del predictor ni el coste de una campaña real. No se midieron energía ni percentiles de latencia en esta selección.

Las siete pruebas nativas con pasos de optimizador pasan en Release y con AddressSanitizer, UndefinedBehaviorSanitizer y detección de fugas. El control nativo de adaptadores pasa también en CUDA. Se reutilizaron los binarios instrumentados y se registraron sus huellas. Estos resultados no cierran la [incidencia de asignaciones CUDA #202](https://github.com/GonxKZ/mars-titan/issues/202).

Cuatro mutaciones dirigidas fueron detectadas: volver a confundir `self` con la clausura, admitir una clausura, eliminar el decaimiento y omitir la comprobación de particiones. Con coverage.py 7.16.1 y Radon 6.0.1, `_apply_decay` cubre sus 6 sentencias y 2 ramas, con CRAP 9. `retained_partitions` cubre sus 8 sentencias y 6 ramas, con CRAP 5. CRAP usa `C² × (1 − cobertura de sentencias)³ + C`. Estas cifras describen recorridos y complejidad, no demuestran ausencia de errores.

## Reproducción y alcance

Se usaron Python 3.12.14, PyTorch 2.14.0+cu130, CUDA 13.0, XGBoost 3.3.0, CuPy 14.2.0 y una RTX 4070 Laptop de 8 GiB. El perfil energético permaneció en `power-saver`. Se limitaron los procesos Python a dos hilos, las pruebas a 6 GiB de RAM y la ejecución propia de GPU a una sola ranura. `hmmlearn==0.3.3` quedó fijado en el extra `reinforcement`, sin cambiar ninguna versión ya presente en `uv.lock`.

Los registros de cada tanda proceden de `python -m pytest -q -rs --durations=8`, lanzado mediante `uv run --no-sync --with hmmlearn==0.3.3` en el entorno aislado. Los identificadores exactos están en `original_cases` y `added_cases` del recibo. El ajuste de parámetros de estos fixtures necesita una política técnica local que lo permita. Estas órdenes no cambian por sí solas la protección global:

```bash
uv run --locked --all-extras pytest -q -rs tests/training/test_anchored_decay.py tests/training/test_anchored_decay_steps.py
uv run --locked --all-extras pytest -q -rs tests/training/test_tabular_retention.py tests/training/test_external_shared_cuda.py
ctest --test-dir build/native/native-controls-release -L optimizer-steps -j 1 --output-on-failure
```

Ruff y la comprobación del repositorio pasan. Los logs, XML, estados de los fixtures, cobertura completa y registros de mutación se conservan fuera de Git. Las huellas del recibo permiten distinguir los intentos fallidos, las regresiones posteriores y los controles instrumentados.

Esta verificación comprueba mecanismos y software. No mide mejora predictiva sobre FinMultiTime, superioridad frente a otras arquitecturas ni novedad científica.
