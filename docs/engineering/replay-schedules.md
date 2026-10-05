# Calendarios de replay con exposiciones emparejadas

`ReplaySchedule` prepara índices C++20 sobre un multiconjunto fijo de episodios maduros. No contiene modalidades ni copia observaciones. Cada episodio declara una identidad, instante observado, disponibilidad de su etiqueta y número de exposiciones. Una fila cuya etiqueta aún no está disponible en el corte hace fallar la preparación.

Los tres controles consumen exactamente esas multiplicidades, también cuando son desiguales. Comparten tamaño de lote y número de actualizaciones, incluida la última fracción de lote. Cambian únicamente el orden:

- `uniform` baraja todo el multiconjunto con Fisher–Yates y extracción entera sin sesgo de módulo.
- `recent` coloca primero las etiquetas disponibles más recientemente, con las exposiciones del episodio contiguas. En caso de empate decide la identidad del episodio.
- `spaced` exige una distancia mínima entre dos exposiciones del mismo episodio, medida en posiciones del calendario. Usa una cola de prioridad por exposiciones pendientes y desempates obtenidos del generador propio.

Si la multiplicidad máxima es m, aparece en k episodios y se solicita distancia d, se necesita `(m - 1) * d + k <= N`, donde N es el total de exposiciones. El control rechaza el caso imposible sin añadir huecos, exposiciones ni actualizaciones. Con multiplicidades iguales puede fijarse d al número de episodios para obtener rondas completas. El espaciado no promete esa distancia cuando las multiplicidades son distintas.

## Estado y confirmación

`prepare()` devuelve una vista de índices `uint32_t` y un token con identidad de ejecución, orden, semilla, cursor, tamaño del lote y contador de actualización. La vista pertenece al calendario y permanece válida mientras este exista. `commit(token)` avanza solo si corresponde al siguiente lote. Un segundo intento con el mismo token se rechaza, igual que la preparación después del último lote.

La identidad de ejecución pertenece al replay intencional y debe identificar ese plan. No es la identidad del feedback aplicado una vez. El consumidor confirma su propio estado junto con el cursor del calendario bajo una barrera de recuperación. El calendario no puede deshacer por sí mismo una actualización del modelo realizada antes de un corte. Recuperar el snapshot previo repite el lote pendiente sobre el estado previo del consumidor. Recuperar el posterior empieza en el lote siguiente.

El snapshot conserva configuración, identidades y multiplicidades, semilla, cursor confirmado y actualizaciones. La construcción consume exclusivamente un `mt19937_64` local. Después no se vuelve a sortear. Restaurar reconstruye el mismo plan y verifica todos los campos y los episodios, sin depender de un generador global ni de la biblioteca que implemente `uniform_int_distribution`. Una versión del formato distingue el algoritmo de calendario. Cambiarlo exige otra versión.

La representación residente de índices ocupa cuatro bytes por exposición. Se admiten como máximo 2²⁰ episodios, 2²⁴ exposiciones y 512 MiB de presupuesto declarado. La comprobación previa reserva margen para metadatos y buffers acotados, además de los índices. No es una medición del overhead del asignador. La lectura del snapshot tiene un máximo de 128 MiB y valida límites antes de reservar sus filas. La construcción uniforme y reciente cuesta O(N + E log E), incluyendo la comprobación de identidades. La espaciada cuesta O(N log E). El consumo recorre N índices.

## Ejecutable y comprobación

El objetivo `mars-titan-replay-control` compara una regresión escalar definida en el ejecutable. La mitad antigua tiene objetivo −1 y la reciente +1. En cada lote aplica `w += 0.03 * (media_objetivo - w)`. Este control ilustra el efecto del orden sobre un estado compartido, sin medir retención predictiva en un mercado.

```sh
cmake -S native -B build/learning-release -G Ninja \
  -DCMAKE_C_COMPILER=clang-21 -DCMAKE_CXX_COMPILER=clang++-21 \
  -DCMAKE_BUILD_TYPE=Release -DMARS_TITAN_BUILD_SIMULATION=OFF \
  -DMARS_TITAN_BUILD_LEARNING_CONTROLS=ON -DMARS_TITAN_WARNINGS_AS_ERRORS=ON
cmake --build build/learning-release --parallel 1
ctest --test-dir build/learning-release --output-on-failure
build/learning-release/mars-titan-replay-control 8192 4 64 71 7
```

Los argumentos son episodios, exposiciones por episodio, lote, semilla y repeticiones. La salida JSON separa construcción y consumo. Hay un calentamiento por calendario. El tiempo interno excluye la serialización del snapshot. El tiempo de proceso externo sí la incluye. Los perfiles de diagnóstico se configuran por separado con `MARS_TITAN_SANITIZER=address-undefined`, `MARS_TITAN_ENABLE_COVERAGE=ON` o `MARS_TITAN_ENABLE_STATIC_ANALYZER=ON`. El objetivo de análisis es `learning-controls-analysis`.

El 5 de octubre de 2026, en un Ryzen 9 8945HS, Clang 21.1.8 y Release, 8.192 episodios con cuatro exposiciones y lote 64 generaron 512 actualizaciones en cada calendario. Los índices ocuparon 131.072 bytes. En siete repeticiones, la mediana de construcción y consumo fue 0,352 ms para uniforme, 0,220 ms para reciente y 0,906 ms para espaciado. El proceso completo de los tres controles, sus calentamientos y snapshots duró 0,04 s y alcanzó 6.056 KiB de RSS según GNU time. La precisión del reloj externo es insuficiente para comparar diferencias pequeñas de proceso. No se detuvieron otros procesos ni se midieron energía o coste monetario.

El orden reciente termina con los episodios antiguos, por su definición de prioridad al principio del calendario. En este control su MAE sobre el objetivo reciente fue 1,9992, frente a 0,9973 del uniforme y 0,9995 del espaciado. No demuestra que uno de los calendarios sea mejor para aprendizaje real.

Las tres pruebas CTest pasaron en Debug y con ASan y UBSan, con detección de fugas activada. El análisis de rutas de Clang no emitió diagnósticos y Lifetime Safety experimental estuvo activado. La cobertura LLVM del planificador fue 98,60 % de líneas y 81,25 % de ramas. Lizard registró CCN máxima 17 en la validación de entradas. CRAP, calculado como `CCN² * (1 - cobertura)³ + CCN` con líneas ejecutables LLVM por función, tuvo máximo 17. Es evidencia diagnóstica, no una prueba de ausencia de errores. El parser completó 1.000 entradas de libFuzzer, con semilla 71, máximo de 4.096 bytes y ASan y UBSan. Las mutaciones que omitían maduración, identidad de episodios y comprobación del token fueron detectadas. `clang-tidy` no estaba instalado. Los valores y límites de la medición están en [el registro del control](../../reports/engineering/replay-schedules-20261005.json).
