# Diagnóstico de la señal de mediana y colas

El diagnóstico no identifica una causa dominante del error de Titans-MAC. En la representación final hay oposición entre contribuciones de gradiente, pero la suma nunca invierte la dirección de la mediana en las filas observadas. Esto no mide el gradiente de todos los parámetros ni descarta interferencia en otros puntos de la trayectoria.

Se repitió únicamente la validación de la referencia original del piloto de claves, con 8.000 etiquetas maduras de 125 sesiones de 2022 y dos instantáneas: inicial y seleccionada después de una época. Se conservaron el calentamiento, los 64 activos y las máscaras. No se aplicó ningún paso de optimizador ni se abrieron calibración, desarrollo de 2023 o el test final. Los parámetros y generadores aleatorios permanecieron iguales. La repetición seleccionada reproduce exactamente sus predicciones confirmadas y ambas reproducciones coinciden con las métricas originales.

La función [`feedback`](../../../src/mars_titan/evaluation/quantile_feedback.py) calcula tres derivadas sobre la representación que entra en la cabeza lineal: la del término principal de mediana, la de los otros cuatro cuantiles y la de la pérdida total. El término principal es L1/10 porque la pinball de la mediana equivale a L1/2 y la pérdida común promedia cinco niveles. No se compara una L1 sin escalar con una contribución escalada. Se usan autodiferenciación y FP32 sobre las representaciones realmente emitidas, sin invertir cuantiles para reconstruir activaciones.

| Instantánea | Filas con oposición local de colas | Filas con inversión del gradiente total | Mediana de norma auxiliar / principal |
| --- | ---: | ---: | ---: |
| Inicial | 25,0625 % | 0 % | 0,157942 |
| Seleccionada | 62,5125 % | 0 % | 0,057884 |

La oposición usa un coseno menor que −0,000001 y la inversión compara el producto con el gradiente total con la misma tolerancia relativa. Las proporciones se promedian primero por sesión. Las dos contribuciones tienen la misma reducción de pérdida. En cada instantánea se capturaron también las 64 predicciones finales pendientes, que no entraron en las 8.000 filas con etiquetas maduras ni en el cálculo de gradientes.

El [recibo](diagnostic.json) conserva las sesiones, normas, tiempos, huellas y límites. La primera reconstrucción usó la identidad de entradas de la vista de validación y el checkpoint rechazó su carga. La repetición recuperó la identidad de entrenamiento guardada en el checkpoint, sin modificarlo ni relajar el contrato. Ese fallo y el control intermedio se conservaron fuera de Git. No se presentan como experimentos predictivos nuevos.

Las [99 comprobaciones focalizadas](verification.json) incluyen CPU/CUDA, FP32/FP64, derivadas calculadas a mano, diferencias finitas, conservación de RNG y rechazo de overflow en la proyección o en el gradiente. El [control de mutación](mutations.json) detectó los seis mutantes definidos y pasó al restaurar el código. La [cobertura y CRAP](quality.json) usan coverage.py y Radon con la convención registrada. Las 25 sentencias y ocho ramas del módulo están cubiertas. Esto no equivale a comprobar todo el repositorio.

El [inventario local](artifacts.json) identifica la captura por muestra y el script de repetición, conservados con los artefactos fuera de Git. La protección global quedó intacta y las autorizaciones locales del diagnóstico se cerraron. La campaña completa permanece bloqueada.
