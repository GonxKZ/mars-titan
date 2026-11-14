# LightGBM en GPU y validación sin repetir árboles

Medición del 14 de noviembre de 2026 para [#847](https://github.com/GonxKZ/mars-titan/issues/847), sobre el origen 2019 de la campaña #836. Se usan las mismas 246.619 filas de ajuste, 128.898 de validación, 1.783 características, semilla 42 y 1.000 rondas. La ventana de ajuste sigue siendo de doce meses. No se utiliza calibración, evaluación ni el test final.

La ejecución anterior llamaba a `predict` sobre todo el prefijo de árboles cada diez rondas. En la primera pasada GPU, esas llamadas consumieron 510 de los 709 segundos de ajuste con validación. El nuevo modo utiliza `valid_sets` y `feval` de LightGBM para leer la predicción que mantiene su validador nativo. Reutiliza los bins del ajuste y evita recalcular cada prefijo. Los puntos de control, las máscaras, el IC y la regla de selección no cambian. Las etiquetas de validación no intervienen en el ajuste.

| Ejecutor | Validación | Preparación (s) | Ajuste con validación (s) | Suma (s) |
| --- | --- | ---: | ---: | ---: |
| CPU | Prefijo completo repetido | 74,20 | 953,85 | 1.028,05 |
| GPU OpenCL float64 | Prefijo completo repetido, segunda pasada | 72,60 | 692,20 | 764,80 |
| CPU | Caché nativa | 75,46 | 335,08 | 410,54 |
| GPU OpenCL float64 | Caché nativa | 75,69 | 145,92 | 221,60 |

La combinación elegida reduce ese recorrido un 78,4 % frente al anterior y tarda 1,85 veces menos que CPU con la misma optimización de validación. Es una medida de un origen y una semilla, no una garantía sobre toda la campaña. La primera pasada GPU, que también se conserva en [measurements.json](measurements.json), tuvo otra configuración de lectura y no entra en esta tabla emparejada.

La caché y la evaluación directa del mismo modelo coinciden exactamente en los valores comprobados: 2.048 filas en los 100 puntos de control y las 128.898 filas completas en las rondas 10, 100, 500 y 1.000, tanto en CPU como en GPU. El control CPU optimizado también reproduce la predicción CPU anterior al final de las 1.000 rondas.

GPU y CPU no producen exactamente el mismo modelo. En la comparación de validación final, 77 filas superan `atol=1e-6` y `rtol=1e-5`. El máximo error absoluto entre predicciones es 0,013775 y su media 0,000003359, en la escala del control normalizado. Esto no es una medida de mejora predictiva. Por eso `lightgbm_gpu` tiene identidad propia y los resultados del control CPU anterior se conservan aparte.

El nuevo control usa OpenCL, doble precisión y la caché nativa. No reduce `max_bin` ni la precisión para conseguir la aceleración. La CPU prepara los datos, gestiona el árbol y mantiene la validación, mientras la GPU calcula los histogramas. La [guía oficial](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html) distingue esta ruta de la implementación CUDA. La opción `deterministic` solo corresponde a CPU, como especifica la [documentación de parámetros](https://lightgbm.readthedocs.io/en/stable/Parameters.html). La ejecución registra los parámetros efectivos.

El pico RSS observado del nuevo control GPU fue de unos 10,18 GiB. El máximo GPU muestreado fue de 538 MiB y contiene el contexto del proceso. No es el contador de asignaciones de PyTorch ni un máximo continuo. La energía no se midió. El recibo conserva las condiciones, los tiempos de comprobación, la procedencia y las limitaciones.

La continuación se prepara en otro estado. Los resultados completados se enlazan como referencias locales y los estados pausados se copian. Se comprueban datos, semillas, recetas, código numérico y versiones. El estado de origen se conserva y su permiso se cierra. La procedencia de cada trabajo e intento acompaña al ZIP de resultados. El segundo ordenador, que todavía no había entrenado, puede empezar directamente con esta versión.

El [recibo de comprobaciones](verification.json) registra 387 pruebas dirigidas, 32 comprobaciones CUDA y una batería general de 14.443 pruebas superadas, 401 omisiones y 16 avisos. Las seis mutaciones de precisión, ausencia de GPU, repetición de validación, código heredado, presupuesto y cierre del permiso de origen fueron detectadas con controles correctos antes y después. La cobertura de seis módulos registra 1.115 de 1.234 sentencias y 339 de 400 ramas con coverage.py 7.16.1 y Radon 6.0.1. CRAP usa C² × (1 − cobertura de sentencias)³ + C. Los recuentos se solapan y no demuestran ausencia de errores.
