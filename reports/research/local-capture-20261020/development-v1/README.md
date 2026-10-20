# Captura local frente a captura global y constante

El contraste predeclarado completó 15 ajustes, tres configuraciones y cinco semillas, sobre las mismas 249 sesiones de desarrollo. La captura local empeora el MAE medio un 7,20 % frente a la global. También pierde frente al control constante y las tres configuraciones pierden frente al retorno residual cero. La variante se conserva desactivada. Este resultado no respalda una mejora predictiva del mecanismo.

| Configuración | MAE medio por sesión | Desviación muestral entre semillas |
| --- | ---: | ---: |
| Captura global | 0,027545532 | 0,008936502 |
| Captura local | 0,029528235 | 0,011096702 |
| Captura constante de 0,5 | 0,023919550 | 0,006832333 |
| Residual cero | 0,015849451 | No depende de la semilla |

La diferencia principal `MAE(local) − MAE(global)` es +0,001982704. El intervalo temporal del 95 % va de +0,000757315 a +0,003231229. Las sensibilidades predefinidas a bloques de 8 y 32 sesiones conservan el signo positivo. El margen relevante fijado era 0,00001. La interpretación es mayor error en este desarrollo para el promedio fijo de estas semillas.

La variabilidad entre semillas es amplia. La local mejora en las semillas 42 y 45 y empeora en 43, 44 y 46. La desviación muestral de la diferencia emparejada es 0,008703789. El intervalo temporal no estima nuevas semillas ni convierte las 249 fechas en 1.245 observaciones independientes. Tampoco se evaluó un ensemble de predicciones.

Frente al control constante, la diferencia local es +0,005608685 y la global +0,003625982. Sus intervalos simultáneos del 95 % son positivos. Frente a cero, el exceso de MAE es del 86,30 % para la local, 73,79 % para la global y 50,92 % para la constante. Los cinco secundarios y sus intervalos se conservan en [results.json](results.json), sin elegir el contraste después de ver cuál gana.

El conjunto contiene 64 activos US de fold-018, con todas sus filas admisibles desde 2000 y ausencias explícitas. Se conservaron 203.350 filas de ajuste, 8.000 de validación, 3.968 de calibración y 15.936 de evaluación. Se usaron una época, 685 actualizaciones por ajuste, K=1, banco episódico desactivado, la cabeza original de cinco cuantiles y la misma receta e inicialización entre las tres configuraciones de cada semilla. Los controles de inicialización pasaron. No se exige igualdad de pesos finales ni se presenta un control unitario entrenado que no se ejecutó.

La fuente del productor y del análisis es `2e8fc538c622bdf44fc3c8d17c079e576f97a53e`. El [protocolo](../../../../docs/experiments/local-capture-v1.md) quedó fijado antes de aprender. El bootstrap circular usa 2.000 réplicas, bloques de 16 sesiones y semilla 20261020. Las cinco semillas son 42–46. Esta ventana ya se había utilizado y el estudio sigue siendo un piloto multisemilla de desarrollo reutilizado, sin confirmación final independiente.

Los 15 ajustes consumieron 2.220,473495 segundos externos y 10.275 actualizaciones, sin nuevos intentos fallidos. El acumulado de 104 intentos es 13.793,836751 segundos de los 14.400 declarados. El análisis se registra aparte, con 6,878217 segundos. No se suman otra vez los tiempos internos. Los máximos registrados son 2.138.718.208 bytes de RSS, 267.316.224 bytes CUDA asignados y 297.795.584 bytes reservados. Son costes de esta ejecución, con escritorio y trabajo CPU concurrente, no una medida aislada de aceleración.

Se publican [preparación](prepared.json), [inicializaciones](initialization-controls.json), [costes](cost-summary.json), [análisis](analysis-manifest.json), [cierre](closure.json) e [inventario de 170 artefactos](artifacts.json). Predicciones por muestra, métricas por sesión, checkpoints y registros permanecen fuera de Git, identificados por ruta relativa, bytes y SHA256. Las autorizaciones locales están cerradas y la protección global conserva su hash. La campaña completa y el test final de 2024 siguen cerrados.

La propiedad técnica de evitar la dilución entre filas no permite atribuir utilidad financiera. Esta comparación tampoco identifica qué dinámica interna causa el mayor error. La decisión observada es conservar la referencia y los controles, sin promover la captura local ni modificar retrospectivamente el protocolo para favorecerla.
