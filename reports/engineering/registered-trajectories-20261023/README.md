# Trayectorias propias para propuestas registradas

La implementación del commit `737166c4e53c2e8d91424f802b4cc9604b4c0297` materializa recetas regionales con los pesos del padre y estados nuevos. Cada configuración ejecuta su propia sesión financiera y conserva sus emisiones nativas. El contraste reconstruye esas emisiones antes de calcular pérdidas o recuperar una selección. La [especificación técnica](../../../docs/engineering/registered-trajectories.md) describe entradas, límites y recuperación.

Esta comprobación usa datos de prueba con objetivos manuales. No entrena parámetros compartidos ni mide una mejora predictiva real. La fase E y el test final no se ejecutan. El proponente aprendido sigue pendiente.

La cobertura y el perfil corresponden al commit anterior. La revisión `4b5e9531275f46ea300082a565551b95626a31f8` cambió únicamente ocho mensajes de error, comprobado mediante comparación del AST sin atributos de posición. Después superó de nuevo las tres pruebas de recuperación en CUDA, sin pasos de optimizador. El resumen conserva ambas huellas de código por separado.

## Pruebas y revisión

Se han superado 333 nodos únicos de `tests/proposals`, con rutas CUDA FP64 y transferencia FP32. La recogida final se coteja con los identificadores del [resumen legible por máquina](summary.json). La pasada con cobertura completó 329 casos antes de interrumpirse con SIGINT para respetar el límite por proceso. Los tres casos pendientes y el nuevo caso FP32 se ejecutaron después, con cuatro resultados verdes. El primer proceso terminó con código 2 y el segundo con código 0. No se presenta el primero como una suite completa verde. No hubo pasos de optimizador.

Las pruebas incluyen copia exacta y sin alias, RNG, codec de otra fase, rechazo de cambios fuera de programa/J, emisiones maduras, censuras, exclusiones, etiquetas alteradas, igualdad con sesiones directas y recuperación de interrupciones. También cubren una falsificación coherente de las hojas derivadas, la sustitución del identificador efectivo y el cierre nativo del segundo intento sin recibo derivado. En este último caso se recupera sin nuevos forwards ni cargos y el tiempo de ejecución perdido permanece desconocido. Los temporales del arranque se aceptan solo con nombres, tipo y tamaño acotados.

Se detectaron dos mutaciones dirigidas, con controles verdes antes y después:

- Tratar la hoja derivada como fuente autoritativa, aunque las emisiones nativas sean distintas.
- Omitir la copia de los tensores del padre.

Son dos casos seleccionados y no una puntuación de mutación de todo el proyecto. La primera tentativa del primer caso agotó el límite de 60 segundos después de terminar pytest y antes de salir del proceso. No llegó a ejecutar el mutante y no cuenta como detección. Se conservó ese resultado y se repitió con 90 segundos por fase, sin cambiar la prueba ni la mutación.

El [informe de cobertura y CRAP](quality.json) utiliza coverage 7.16.1 con ramas y Radon 6.0.1. En `candidate_runtime.py`, `trajectory_reader.py`, `trajectory_contrast.py` y `error_rule.py` cubre 764 de 774 sentencias y 128 de 144 ramas. CRAP se calcula como `v² × (1 − c)³ + v`, con complejidad Radon y cobertura de sentencias dentro de cada función. El máximo es 34 en `_read_state`, cuya cobertura de sentencias y ramas es completa. Esta medida no comprueba todas las combinaciones de condiciones booleanas ni demuestra ausencia de defectos.

## Ejecución pequeña sin cobertura

Se ejecutó una única prueba integral adicional en `cuda:0`, con cuatro flujos ficticios, tres sesiones puntuadas y doce predicciones por configuración. Los tramos G/S/E de la fixture abarcan menos de un mes. Usa contexto de 64 sesiones y representaciones reducidas de precios, noticias, gráficos, fundamentales y macro. Las dimensiones son `(64, 5)`, `2`, `1`, `3` y `3`. No son los embeddings ni la cobertura del corpus financiero real.

La referencia utiliza `summary_at_end_v1` y J1. Las otras dos configuraciones cambian, por separado, al programa `summary_interleaved_v1` o a J2. Conservan K1, núcleo MAC, parámetros emparejados, política M1, entradas, etiquetas y calendario. Cada una tiene seis generaciones nativas. La ejecución mantiene la referencia como seleccionada en esta fixture, sin interpretar ese resultado como evidencia financiera.

| Configuración | Construcción y apertura (s) | Ejecución (s) | Lectura y comprobación (s) |
| --- | ---: | ---: | ---: |
| Referencia | 0,237 | 6,839 | 0,491 |
| Programa intercalado | 0,257 | 6,673 | 0,439 |
| Puente J2 | 0,269 | 8,384 | 1,001 |

El contraste completo tardó 25,461 segundos. Preparar la fixture y G tardó 10,510 segundos. El proceso externo completo duró 44,156 segundos, incluidos importación, inventarios y serialización. El programa intercalado registra 32 evaluaciones de resumen frente a 16. J2 registra 24 actualizaciones de nodos del puente frente a 12. Todos mantienen 16 observaciones del núcleo y de MAC, incluyendo calentamiento, y doce refinamientos del lector.

Los picos acumulados del proceso fueron 1 494 102 016 bytes de RSS, 68 465 152 bytes asignados por Torch y 69 206 016 bytes reservados por Torch. Incluyen preparación y todas las configuraciones. No son incrementos exclusivos por configuración ni toda la memoria del contexto CUDA. La GPU tenía otra carga local activa. No se detuvo para medir. La ejecución usa una RTX 4070 Laptop, PyTorch 2.14.0+cu130, CUDA 13.0, dos hilos de cálculo y uno entre operaciones. El modelo es FP64 y la política conserva FP32 estricto para las operaciones de ese tipo.

Una ejecución por configuración no estima percentiles ni acredita una aceleración reproducible. Tampoco se midieron energía ni volumen de transferencias. El primer intento de este perfil falló antes de construir la fixture por una ruta de importación auxiliar de las pruebas. Se corrigió el lanzador local y se conservó el intento fallido, sin atribuirle ejecuciones de las configuraciones.

Los manifiestos locales conservan configuración, código, fuente, predicciones por muestra, estadísticas por sesión, sesiones nativas y sus hashes. El resumen publicado registra sus identidades sin copiar los artefactos voluminosos al repositorio. La evidencia comprueba mecanismos y trazabilidad local. No confirma utilidad predictiva, generalización, especialización aprendida ni novedad científica.
