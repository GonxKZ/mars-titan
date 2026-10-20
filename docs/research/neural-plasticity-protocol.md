# Piloto de plasticidad neuronal

Este protocolo fija la comparación de desarrollo antes de ejecutar nuevos ajustes. La [declaración](../../configs/experiments/neural-plasticity-study-v1.json) y el [catálogo de variantes](../../configs/experiments/neural-plasticity-v1.json) son sus entradas ejecutables. No autoriza la campaña completa ni el test final. El ejecutor conserva el bloqueo por defecto y necesita una excepción local acotada.

La comparación reutiliza la cohorte de 64 acciones de la representación original del estudio de refinamientos. Se conservan todas sus filas, modalidades y máscaras. La vista declara 203.350 filas de ajuste, 8.000 de validación, 3.968 de calibración y 15.936 de desarrollo posterior. Estas cifras describen la vista seleccionada, no la integridad de todo el universo histórico.

| Tramo | Intervalo, con extremo final excluido | Uso |
| --- | --- | --- |
| Ajuste | 1 de enero de 2000 a 1 de abril de 2022 | Una época sobre todas las filas admitidas |
| Validación | 1 de abril a 1 de octubre de 2022 | Selección del checkpoint con MAE por sesión |
| Calibración | 1 de octubre de 2022 a 1 de enero de 2023 | Función de calibración definida por el ejecutor |
| Desarrollo | 1 de enero de 2023 a 1 de enero de 2024 | Comparación emparejada posterior al ajuste |

La purga sigue el intervalo real de cada etiqueta. El calentamiento usa los doce meses anteriores admitidos por la receta. Los estados de cada fase se reconstruyen con el pasado permitido. Se reutiliza el lector cronológico, su tratamiento de etiquetas maduras y su recuperación. El tramo de 2023 ya se ha consultado en estudios anteriores y no es una muestra confirmatoria nueva.

## Contrastes

El contraste principal es `cascade_combined` frente a `cascade_original`. Se conservan también `cascade_write`, `cascade_capture`, las tres combinaciones de ganancias constantes y `cascade_unity`. `mac_original`, `write_coherence` y `fixed_write` permiten examinar la escritura sin añadir PT5. Hay once modelos, cada uno con identidad propia.

Todos parten de los mismos parámetros compartidos para la semilla 42, sin transferir checkpoints seleccionados por resultados anteriores. Se emplean las proyecciones `titans_mac_paper_projections_v2`, dimensión oculta 64, una capa de atención, la misma cabeza de cuantiles y K=1. La memoria episódica permanece desactivada en este contraste. Las actualizaciones internas de la memoria neuronal no cuentan como refinamientos K.

La tasa de aprendizaje externa es 0,001, con una época y presupuesto fijo. La selección conserva el mejor checkpoint según el MAE de validación, con margen de selección 0,00001. No se amplía el ajuste de una variante por observar que pierde. Los parámetros compartidos permanecen congelados durante calibración y desarrollo. La actualización interna neuronal solo utiliza sus entradas asociativas admitidas.

El piloto tiene una semilla. No permite estimar variación entre semillas ni afirmar superioridad general frente al estado del arte. Las configuraciones constantes comprueban si un cambio se explica simplemente por escribir o intercambiar menos. La referencia residual cero se evalúa sobre las mismas observaciones.

## Medición y decisión

La métrica principal es el MAE residual por sesión. Primero se promedian los activos de una misma decisión del mercado estadounidense y después sus sesiones. Se exigirán los mismos identificadores, etiquetas y pesos temporales en todas las comparaciones. Un modelo incompleto no se comparará tras eliminar sus filas ausentes.

Se informará de `Delta_error = MAE_variante − MAE_referencia` y del porcentaje de reducción respecto a la referencia. El porcentaje queda indefinido si su denominador es cero. El margen mínimo de interés del piloto se fija en 0,00001 de MAE. Un efecto menor, un intervalo que cruce cero o una diferencia que no supere ese margen no justifican una mejora respaldada.

El remuestreo conserva las sesiones, con bloques temporales de 16 fechas observadas, 2.000 réplicas, semilla 20261020 e intervalos del 95 %. La sensibilidad usa bloques de 8 y 32. No se remuestrean activos como réplicas temporales independientes. Los contrastes secundarios son exploratorios y se publican todos, sin presentar su mínimo como contraste principal.

La interacción se describe en la escala del error como `MAE_combinada − MAE_escritura − MAE_captura + MAE_PT5`. Una interacción negativa no prueba causalidad económica ni utilidad fuera de esta ventana. Se conservan los controles cero y constantes aunque una comparación entre variantes sea favorable.

El informe recogerá tiempo de cada intento, pasos de optimizador, RAM y VRAM pico, filas procesadas, checkpoints elegidos y coste de inferencia medido. La lectura de eficiencia del piloto mantiene exposición y actualizaciones comparables. No equivale a una comparación con igual tiempo de cómputo.

## Límites y recuperación

El presupuesto acumulado es de 14.400 segundos locales, con 1.800 segundos por intento y hasta 4.096 pasos observados por intento. Se cuenta el coste de fallos e interrupciones. La longitud fija de la época y los contadores del checkpoint permiten contrastar las actualizaciones totales al terminar cada modelo. Una reserva sin cierre verificable impide iniciar otro intento.

La selección incluye todos los modelos declarados. Si el presupuesto no permite completarlos, el informe identificará los pendientes y no sustituirá el contraste por un subconjunto más favorable. Los estados del entrenador conservan pesos, optimizador, RNG, memoria neuronal, cursores y selección conforme al contrato de recuperación existente. El código, las recetas y la vista se sellan antes de ejecutar. Una modificación exige otro manifiesto.

El esquema 2 de preparación separa `input/view.json`, las declaraciones de `recipes/` y las salidas de `runs/`. Los resultados nunca se escriben dentro de una carpeta protegida de entradas. Los intentos fallidos de integración anteriores a cualquier actualización conservan su código, recibo y coste.

No se ejecutan postentrenamientos, políticas RL, búsquedas masivas ni cómputo externo como parte de este piloto. Una mejora de su error sería un resultado de desarrollo. La confirmación multisemilla, otras ventanas y cualquier acceso al test final requieren sus propios protocolos y las condiciones de acceso vigentes.

## Repetición con controles escalares

El piloto v1 terminó, pero su control unitario no reprodujo la referencia después del ajuste. Dos repeticiones con la misma semilla conservaron exactamente las predicciones de cada configuración. Se identificó una diferencia en el orden de reducción del gradiente de la puerta de escritura. Los resultados de v1 se conservan como observaciones de esa implementación y no permiten atribuir al mecanismo las diferencias entre configuraciones.

La [declaración v2](../../configs/experiments/neural-plasticity-study-v2.json) y su [catálogo](../../configs/experiments/neural-plasticity-v2.json) cambian únicamente el contrato numérico de los controles descrito en la documentación de implementación. Conservan las once configuraciones, los datos, los intervalos, la semilla, la época, la tasa de aprendizaje, la selección y el análisis estadístico. No se ajustan esas decisiones a los resultados de v1.

Primero se ejecutan `cascade_original` y `cascade_unity`. Se exige igualdad exacta de los tensores compartidos del checkpoint seleccionado y de las predicciones de validación, calibración y desarrollo. También se comprueba que la inicialización compartida sea la misma. Si falla este control, se detiene la repetición antes de interpretar las demás configuraciones. El análisis conserva las diferencias numéricas y marca bloqueada su atribución.

Los fallos, el piloto v1 y sus repeticiones siguen contando dentro del presupuesto acumulado de 14.400 segundos. No se reinicia el contador con la nueva versión. La preparación se realiza con `uv run python -m benchmarks.neural_plasticity_study prepare --version v2`, junto con las rutas `--view` y `--work`. El ejecutor toma la versión del manifiesto sellado. Preparar ese manifiesto no abre el bloqueo de aprendizaje ni el test final.
