# Selección y recuperación de la edición de convergencia

La edición nueva retrasa el consumo de paciencia hasta completar un mínimo de aprendizaje. La selección del mejor estado comienza desde el principio. Las continuaciones neuronales pueden conservar el padre y los adaptadores, su estado inicial. Alcanzar el techo se registra como presupuesto agotado. La parada por falta de mejora de validación no demuestra un óptimo global ni elimina todo riesgo de sobreajuste.

| Método | Mínimo | Paciencia posterior | Techo | Criterio |
| --- | --- | --- | --- | --- |
| Referencias neuronales | 10 épocas | 10 validaciones completas | 100 épocas | MAE por sesión, mejora mínima de 0,00001. |
| Postentrenamiento real | 5 épocas | 8 validaciones completas | 50 épocas | MAE por sesión, mejora mínima de 0,00001. |
| XGBoost | 200 rondas | 100 validaciones completas | 2.000 rondas | MAE por sesión, mejora mínima de 0,00001. |
| PPO y Double DQN nativos | 131.072 transiciones | 8 validaciones completas | 1.048.576 transiciones | Menos ruinas y después mayor crecimiento logarítmico, con mejora mínima de 0,0001. |

Los controles neuronales pareados de la búsqueda conservan cinco épocas por caso, con 96 controles por edición. Esto incluye la búsqueda nueva. Las comparaciones aumentadas siguen identificadas como estudios de presupuesto fijo. El postentrenamiento real con paciencia es una comparación separada. Ridge mantiene su solución regularizada, sin épocas artificiales. Los datos, recibos y configuraciones anteriores permanecen separados de la nueva edición.

La revisión de las 64 referencias anteriores encontró tres que agotaron las 30 épocas mientras mejoraban en la última validación, tres que llegaron al techo sin mejora en la última época, 24 que pararon antes de completar una ventana de diez épocas más diez evaluaciones de paciencia y 34 que pararon con la paciencia anterior. Estas categorías describen el presupuesto ejecutado. No prueban que entrenar más vaya a mejorar cada caso.

## Recuperación comprobada

XGBoost puede producir otro corte del árbol tras cargar un modelo y reconstruir su caché interna, aunque la diferencia inicial en las predicciones sea pequeña. En la edición nueva, la recuperación reproduce el prefijo desde la misma semilla y datos, comprueba la igualdad binaria del modelo confirmado y continúa con la caché reconstruida. Solo se normalizan dos atributos de auditoría. No se omiten pesos ni configuración del modelo. La reconstrucción no vuelve a publicar rondas antiguas ni consume paciencia. Una pausa durante ese recorrido conserva el checkpoint confirmado.

El coste adicional se concentra en recuperar. En una fixture de 24 filas de entrenamiento, nueve de validación y 325 variables, cinco repeticiones con calentamiento dieron medianas de 162,68 ms antes y 159,65 ms después para cuatro rondas, y de 362,98 y 358,38 ms para 64. Los intervalos se solapan, por lo que no se afirma una aceleración. Reconstruir 32 rondas tardó una mediana de 15,81 ms dentro de una recuperación total de 289,15 ms. Esta medición no estima el coste del corpus real ni de 2.000 rondas.

El ejecutor C++20 guarda los cursores reales de validación. Los entornos pueden terminar su calentamiento en momentos distintos y avanzar con lotes incompletos. Se verificaron pausa, recuperación y auditoría con cursores `0, 31, 32`, sin exigir múltiplos exactos del intervalo. El esquema nuevo limita el historial a 4.096 evaluaciones posibles antes de iniciar el runtime. La configuración principal necesita como máximo 65. Double DQN respeta además su calentamiento antes de consumir paciencia.

El mejor checkpoint y los dos recientes de recuperación tienen funciones distintas. La sustitución confirma el archivo y el índice antes de retirar estados antiguos. Las comprobaciones incluyen archivos corruptos, pausas y recuperación del optimizador, RNG y cursor. Los recibos nativos de esquema 3 se incorporan al observatorio y las métricas reservadas de auditoría siguen fuera de la publicación.

## Verificación local

La rama conjunta pasó 2.616 pruebas Python en 369,30 segundos, sin omisiones, con CUDA, codificadores reales y `hmmlearn` activados. La corrección posterior del contrato público pasó 110 pruebas dirigidas, que se solapan con la suite general. CTest pasó sus 14 pruebas en Release y Node, las 27 del frontend. Ruff, formato y comprobación del repositorio terminaron correctamente.

La comprobación nativa utilizó Clang 21, C++20, avisos como errores, análisis estático y ASan/UBSan. Quince pruebas específicas pasaron con sanitizadores. LibTorch y Arrow del entorno no están instrumentados. No se repitieron TSan ni MSan porque no hay cambios de concurrencia y faltan dependencias completamente instrumentadas para MSan. Se detectaron siete mutaciones dirigidas de boosting, dos del selector supervisado y dos del contrato nativo.

La [evidencia estructurada](convergence-verification-20261001.json) identifica revisiones, versiones, recuentos y límites. La duración de la suite no se utiliza como benchmark. Hubo otra carga CUDA durante parte de la ejecución. Los avisos de deprecación permanecen registrados. Estos resultados comprueban el funcionamiento de la implementación, no una mejora predictiva de la campaña nueva.

El test real de 2024 sigue cerrado. La comparación financiera continúa limitada a mundos sintéticos de contabilidad conocida. El candidato MARS-TITAN sigue sin implementar ni entrenar.
