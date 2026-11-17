# Admisión de RAM y recuperación de la comparación

El 16 de noviembre a las 11:55 de Madrid, el kernel cerró el proceso de Titans-MAC 2023, semilla 43, por agotamiento global de RAM. Su checkpoint confirmado conservaba el paso 685. El grupo de recursos incluía el controlador, que también terminó. Al relanzarlo, el límite común de 12 GiB y los 4 GiB adicionales exigidos para el sistema lo dejaron esperando más de un día con unos 15 GiB disponibles.

Los intentos registrados alcanzaban un máximo RSS de 6,12 GiB en trabajos no tabulares y 10,26 GiB en LightGBM. El nuevo lanzador aplica un límite de 8 GiB a cada trabajo neuronal, Ridge y la regla, y mantiene 12 GiB para LightGBM. El controlador utiliza otro grupo de recursos con un límite de 2 GiB. El margen de admisión sigue siendo de 4 GiB y se conserva una sola carga científica GPU.

El script operativo reutiliza el controlador del runtime congelado `eb01b883e8e6ec912e2cda85f738fd2525584d19`. No cambia datos, recetas, presupuestos de épocas ni módulos numéricos. Comprueba el permiso de esa versión y devuelve los errores al controlador para que registre los intentos y aplique sus reintentos acotados. `resource-status.json` diferencia la espera de recursos del proceso de entrenamiento real.

## Comprobaciones

Las 26 pruebas propias y las 39 del supervisor pasan, sin omisiones. Comprueban límites y márgenes, espera, parada antes de lanzar, recuperación del estado de un proceso al recibir la parada, salida del padre, propagación de errores, identidad del runtime y exportación solo cuando la cola está completa. Ruff, formato, comprobación del repositorio y revisión de espacios pasan.

Se detectaron seis mutaciones dirigidas. En la primera pasada sobrevivió omitir la comprobación de parada justo antes del lanzamiento. La prueba anterior solo comprobaba que no se escribiera el archivo del trabajo. Se amplió para detectar también la creación de un proceso innecesario, y la mutación falla con esa prueba. Los controles sin mutar siguen pasando. La revisión de la CLI añadió una regresión que rechaza un lanzador no ejecutable antes de construir la cola. Los [recibos](mutations.json) conservan los códigos de salida finales.

La [cobertura](quality.json), medida con coverage.py 7.16.1 y complejidad de Radon 6.0.1, registra 158/184 sentencias y 49/62 ramas. El CRAP máximo observado es 25,79 en `run_worker`, con cobertura de sentencias. Estas cifras corresponden a las pruebas CPU, no a la recuperación real CUDA. Quedan ramas de error y de la entrada CLI sin cobertura instrumentada.

## Recuperación observada

La cola volvió a arrancar el 17 de noviembre a las 20:51 de Madrid. Se comprobó CUDA y la carga íntegra del checkpoint del paso 685 antes de iniciar el trabajo. A las 20:57, Titans-MAC había confirmado el paso 1.197. Los 60 trabajos ya completos se conservaron.

El [recibo de recuperación](recovery.json) registra el límite real del trabajo, 8 GiB, un pico de 6,66 GiB contabilizado por su grupo de recursos y 11,42 GiB de RAM disponible en esa consulta. La contabilidad del grupo incluye memoria distinta del RSS y no se compara como si fuera el mismo contador. El controlador quedó en un grupo separado del trabajo.

La mejora comprobada consiste en retirar una espera que impedía ejecutar trabajo útil. No se ha medido una aceleración de las operaciones del modelo ni se afirma máxima eficiencia del hardware. Los picos anteriores no garantizan que todas las variantes pendientes quepan. El límite mayor de LightGBM puede volver a exigir espera si no hay RAM suficiente. La reserva de admisión no impide que otra aplicación consuma memoria después.

La [verificación](verification.json) reúne los recuentos y límites de esta revisión. No aporta resultados predictivos nuevos ni abre el test final.
