# Registro finito de configuraciones financieras

`src/mars_titan/proposals/contracts.py` implementa `RegisteredConfiguration` y `ProposalRegistry`. Identifica configuraciones del predictor y del lector, con acción, catálogo y precisión explícitos. Admite hasta ocho entradas y exige una referencia incluida en el registro. Conserva FP32/FP64, acción fija y ausencia de calibrador y selector aprendido. Los sellos se vuelven a comprobar al consultar las identidades o resolver una configuración. J y K mantienen contratos distintos.

La [verificación final](verification.json) recoge 27 casos únicos sobre metadatos y configuraciones CPU. No construyen predictores ni ejecutan forwards o pasos de optimizador. El preflight CUDA previo comprueba disponibilidad, pero no convierte estos casos en 27 pruebas CUDA.

Se conservan el RED inicial de un caso por ausencia del módulo y los dos fallos `DID NOT RAISE` al alterar tipos internos con JSON equivalente. La versión final rechaza esos cambios sin renovar el sello.

El [resumen de calidad](quality.json) registra 93/93 sentencias y 28/28 ramas, 13 funciones y CRAP máximo de 20. Se usaron coverage.py 7.16.1 y Radon 6.0.1. El alcance es el paquete del registro, cuyo `__init__.py` no tiene sentencias medibles. Estas cifras no prueban todas las combinaciones posibles ni ausencia de errores.

Se detectaron [cuatro objetivos dirigidos](mutations.json): omitir el sello, retirar la compatibilidad del lector, admitir nueve entradas y omitir la validación de tipos al consultar la identidad. Sus 12 fases incluyen referencia y restauración correctas y un fallo semántico en CALL por mutante. No se han comprobado todas las mutaciones posibles. El intento V1 del sello se conserva como no aprobado por el formato del informe de pytest y no cuenta como detección. Los tiempos de los recibos son de verificación, no medidas de rendimiento.

Este registro no genera propuestas desde resultados maduros, compara candidatas, aprende una política ni despliega consumidores. No completa el ciclo exterior de #711 ni acredita mejora predictiva. La campaña y el test final siguen cerrados.
