# Piloto de centrado y competencia de claves

La combinación de centrado y competencia empeora frente a Titans-MAC original en este piloto. La competencia sin centrado tiene un MAE algo menor que la referencia, con una diferencia inconcluyente. Todas las configuraciones pierden frente al residual cero. No se promueve ninguna variante.

Se completaron las siete configuraciones del [protocolo](../../../../docs/experiments/key-competition-v1.md), con semilla 42 y una época. El código quedó congelado en `43308173bf79e59b9d7d5aadda06b5b18af6088a`. Cada modelo hizo 685 actualizaciones con 203.350 filas de ajuste. Se conservaron las mismas 8.000 filas de validación, 3.968 de calibración y 15.936 de desarrollo, estas últimas correspondientes a 249 sesiones de 2023. Son 64 activos estadounidenses con todas sus filas admisibles desde 2000 y sus máscaras. No es una evaluación del universo completo. El test final de 2024 siguió cerrado.

## Resultado predictivo

| Configuración | MAE residual por sesión |
| --- | ---: |
| Residual cero | 0,01584945 |
| Original | 0,01747972 |
| Todas las componentes | 0,01747972 |
| Centrado | 0,02405764 |
| Competencia | 0,01724874 |
| Centrado y competencia | 0,02377246 |
| Coordenadas fijas | 0,02451098 |
| Centrado y coordenadas fijas | 0,01787978 |

El contraste principal, combinación menos original, es +0,00629274, con intervalo temporal del 95 % [+0,00503037, +0,00759362]. La mejora relativa es −36,00 %, es decir, el error aumenta un 36,00 %. La hipótesis de beneficio de la combinación no recibe apoyo en esta ventana y semilla.

La competencia sin centrado tiene una diferencia de −0,00023098 frente al original. Su intervalo simultáneo del 95 % es [−0,00065385, +0,00019189], por lo que es inconcluyente. Frente a cero, esa misma configuración tiene una diferencia positiva de +0,00139929, con intervalo simultáneo [+0,00113617, +0,00166241]. La ligera reducción puntual frente a la referencia no basta para elegirla como arquitectura.

La competencia mejora frente a coordenadas fijas sin centrado, con diferencia −0,00726223 e intervalo simultáneo [−0,00870118, −0,00582329]. Con centrado, la competencia empeora frente a coordenadas fijas: +0,00589268, intervalo [+0,00372046, +0,00806489]. Los controles no respaldan un beneficio uniforme de seleccionar por magnitud. No se identifica el mecanismo que produce estas diferencias solo a partir del MAE.

La interacción `MAE_combinada − MAE_centrada − MAE_competitiva + MAE_original` es −0,00005420, con intervalo simultáneo [−0,00167402, +0,00156562]. Es inconcluyente y no demuestra sinergia. Los intervalos usan bloques temporales de 16 sesiones, 2.000 réplicas y sensibilidad predefinida a 8 y 32. El [resultado completo](results.json) conserva las métricas adicionales, los contrastes y sus sensibilidades. No se ajustó una nueva calibración para favorecer las variantes.

La evaluación reutiliza desarrollo ya estudiado y una sola semilla. No estima la variación entre semillas ni acredita generalización a otras ventanas, activos o mercados. No demuestra superioridad frente al estado del arte. La motivación neurocientífica no proporciona una garantía predictiva y los [contraejemplos](../../../../docs/research/key-competition.md#alcance-matemático-y-objeciones) ya descartan una separación uniforme de patrones.

## Paridad, coste y conservación

El [control completo](unity-control.json) reproduce bit a bit los 45 tensores compartidos y todas las predicciones de validación, calibración y desarrollo después de 685 actualizaciones. La inicialización coincide. La [comparación con la referencia histórica](baseline-historical-parity.json) confirma también los mismos parámetros y predicciones que `mac_original` del piloto de plasticidad v2. No se eligió retrospectivamente una referencia más débil.

Los siete trabajos consumieron 871,602 segundos externos, unos 14,53 minutos, con 4.795 actualizaciones en total. Los 33 intentos acumulados, incluidos los 26 anteriores, suman 4.313,545 segundos externos. El análisis consumió otros 4,425 segundos de proceso, con 0,862 segundos dentro del analizador. Los tiempos internos están incluidos en los externos y no se suman dos veces. Estas cuentas no incluyen toda la investigación, las pruebas o la medida técnica de costes, que registra por separado 40,895 segundos.

El mayor RSS de los siete trabajos fue de 2.118.119.424 bytes. PyTorch registró un máximo de 206.389.248 bytes asignados y 236.978.176 reservados en CUDA. Se utilizó una RTX 4070 Laptop de 8 GiB, PyTorch 2.14.0+cu130, FP32 estricto y una sola carga científica de GPU. Hubo tareas breves de documentación y comprobación en CPU durante algunos ajustes. No son tiempos medidos con la máquina inactiva.

Las variantes no añaden parámetros ni estado persistente. La [medida técnica](../README.md#coste-cuda-con-entradas-precodificadas) mantiene 186.823 parámetros y 67.856 bytes tensoriales por flujo. La combinación aumenta la mediana de inferencia y no aporta ahorro persistente. Los pesos rápidos siguen siendo densos. No se midieron energía ni el pico total del dispositivo.

El [manifiesto preparado](prepared.json) fija fuentes, recetas, entorno y población. El [inventario](artifacts.json) identifica las predicciones por muestra, checkpoints, manifiestos y métricas por sesión conservados fuera de Git. El [cierre](closure.json) verifica todos los intentos y autorizaciones locales cerrados, con la protección global intacta. No queda un entrenamiento ejecutándose en segundo plano.

Se conservan implementación y resultados negativos con las opciones desactivadas. No se amplía este piloto a una campaña confirmatoria ni se cambian sus controles para obtener un resultado favorable.
