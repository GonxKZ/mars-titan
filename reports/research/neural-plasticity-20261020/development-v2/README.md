# Plasticidad neuronal con controles escalares

El piloto no respalda promover la escritura por coherencia ni la captura por elegibilidad. La combinación reduce el error frente a PT5, pero no mejora de forma concluyente frente a dos ganancias constantes. Las dos modulaciones individuales pierden frente a sus respectivos controles constantes. Todas las configuraciones tienen más error que el residual cero.

Se ejecutaron las once configuraciones del [protocolo](../../../../docs/research/neural-plasticity-protocol.md) con `scalar_controls_v2`, semilla 42 y una época. El código quedó congelado en `ff755cde2f7ce3c20d4298c0d5ccd85aa418143c`. Cada modelo hizo 685 actualizaciones con 203.350 filas de ajuste. Se conservaron las mismas 8.000 filas de validación, 3.968 de calibración y 15.936 de desarrollo, correspondientes estas últimas a 249 sesiones de 2023.

La preparación usa 64 activos estadounidenses y la edición histórica con máscaras. Esa cohorte acotada no verifica el universo completo desde 2000. El test final de 2024 y la campaña completa permanecieron cerrados.

## Resultado predictivo

| Configuración | MAE residual por sesión |
| --- | ---: |
| Residual cero | 0,01584945 |
| Titans-MAC original | 0,01747972 |
| Escritura constante sin PT5 | 0,01685220 |
| Escritura por coherencia sin PT5 | 0,01985341 |
| PT5 original | 0,02547209 |
| PT5 con ganancias unitarias | 0,02547209 |
| PT5 con escritura modulada | 0,02349177 |
| PT5 con captura modulada | 0,02119058 |
| PT5 con ambas modulaciones | 0,02217319 |
| PT5 con escritura constante | 0,02066987 |
| PT5 con captura constante | 0,01946323 |
| PT5 con ambas ganancias constantes | 0,02201881 |

El contraste principal, combinación menos PT5, es −0,00329890, con intervalo temporal del 95 % de [−0,00511559, −0,00149755]. La reducción relativa es del 12,95 %. El resultado pertenece a esta ventana y semilla. No acredita superar a Titans-MAC original, al residual cero ni al estado del arte.

Frente a sus controles constantes, las diferencias exploratorias de MAE son:

| Comparación | Diferencia | Intervalo simultáneo del 95 % |
| --- | ---: | --- |
| Coherencia sin PT5 menos escritura constante | +0,00300120 | [+0,00239409, +0,00360832] |
| PT5 con escritura modulada menos escritura constante | +0,00282190 | [+0,00128961, +0,00435419] |
| PT5 con captura modulada menos captura constante | +0,00172736 | [+0,00089571, +0,00255900] |
| PT5 con ambas modulaciones menos ambas constantes | +0,00015439 | [−0,00076046, +0,00106924] |

La interacción `MAE_combinada−MAE_escritura−MAE_captura+MAE_PT5` es +0,00296293, con intervalo simultáneo [+0,00077379, +0,00515206]. No apoya una reducción adicional del error al combinar las modulaciones. Tampoco es una prueba de causalidad económica.

Los intervalos usan bloques temporales de 16 fechas, 2.000 réplicas y sensibilidad preespecificada a 8 y 32 fechas. Las filas, configuraciones y repeticiones de control no se cuentan como nuevas observaciones temporales. El [resultado completo](results.json) conserva MSE/RMSE, dirección, rangos, cuantiles brutos, sensibilidad e intervalos de cada contraste. No se añadió una calibración posterior para favorecer estas variantes.

## Validez del control y coste

El [control unitario](unity-control.json) reprodujo exactamente los 45 tensores compartidos y todas las predicciones de validación, calibración y desarrollo. La inicialización también coincide. Este control se ejecutó antes de las otras nueve configuraciones. La [versión v1](../development-v1/README.md) conserva sus resultados y el fallo de paridad que motivó la corrección.

Las once ejecuciones v2 consumieron 1.628,178 segundos externos, unos 27,14 minutos. Los 26 intentos acumulados, incluidos fallos, v1 y repeticiones de diagnóstico, suman 3.441,943 segundos. Los tiempos internos forman parte de los externos. El análisis final instrumentado consumió otros 4,575 segundos de proceso. Estas cifras no incluyen toda la investigación ni las pruebas técnicas.

El mayor RSS de v2 fue de 2.117.525.504 bytes. El máximo asignado por PyTorch fue de 267.552.768 bytes y el máximo reservado de 295.698.432 bytes. Se utilizó una RTX 4070 Laptop de 8 GiB con PyTorch 2.14.0+cu130, precisión FP32 y una carga científica de GPU simultánea. Las comprobaciones y tareas de documentación en CPU pudieron coincidir con el ajuste. Estos tiempos no son una comparación de velocidad en una máquina inactiva.

La modulación añade 33.792 bytes persistentes por flujo con dimensión 64 y dos capas de memoria. Los controles constantes conservan ese estado auxiliar. Esa cuenta excluye el grafo temporal, temporales y memoria de las bibliotecas. No se midieron energía ni el pico total del dispositivo.

El [manifiesto preparado](prepared.json) fija configuraciones, fuentes, entorno y población. El [inventario de artefactos](artifacts.json) identifica las predicciones por muestra y las métricas por sesión conservadas fuera de Git. Se mantuvo el bloqueo global y se cerraron las excepciones locales al terminar.

## Decisión

Se conservan las variantes implementadas y desactivadas. La evidencia favorece investigar primero alternativas sencillas de escala de escritura y acoplamiento antes de ampliar esta modulación. El piloto no confirma las hipótesis de beneficio adaptativo ni justifica una selección de arquitectura para la campaña.

La incertidumbre entre semillas, otras ventanas, la retención de asociaciones raras y la respuesta a cambios siguen sin evaluarse aquí. Los [contraejemplos del mecanismo](../../../../docs/research/neural-plasticity.md) prueban límites de la interpretación de coherencia y captura, no efectos financieros. No se prolongó el ajuste ni se cambiaron los controles para obtener un resultado favorable.
