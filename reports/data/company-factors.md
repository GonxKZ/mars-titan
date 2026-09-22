# Factores empresariales y comprobación multimodal

Fecha de ejecución: 22 de septiembre de 2026.

La ampliación incorpora siete ratios de balance al vector contable. Conserva
las 65 muestras del conjunto editorial revisado, 30 de MNST y 35 de DECK.
ABM y CSGS siguen sin muestras admitidas. No se ha ampliado la población
entrenable aceptando noticias sin contrastar.

La versión final del [registro de materialización](company-factor-materialization-v3.json)
contiene las fórmulas, los hashes y el orden de los 15 conceptos. La
[versión anterior](company-factor-materialization-v2.json) se conserva como
registro de los manifiestos usados por los entrenamientos. Los Parquet de
muestras de ambas versiones tienen exactamente las mismas huellas.

## Cobertura comprobada

| Factor | MNST, 30 decisiones | DECK, 35 decisiones |
| --- | ---: | ---: |
| Liquidez corriente | 30 | 35 |
| Fondo de maniobra / activo | 30 | 35 |
| Pasivo / activo | 0 | 0 |
| Patrimonio / activo | 30 | 35 |
| Cuentas por cobrar / activo | 30 | 35 |
| Cuentas por pagar / activo | 30 | 0 |
| Pasivo / patrimonio positivo | 0 | 0 |

Son decisiones multimodales, no publicaciones distintas ni una estimación del
universo completo. Se conserva la ausencia del pasivo total cuando no está el
concepto exigido. No se reconstruye restando partidas ni se equipara a deuda.
Todos los componentes originales del vector contable y las otras entradas
coinciden exactamente con la representación de 24 canales.

Las dos particiones no vacías crecen de 302.367 a 303.911 bytes de Parquet.
El vector añade 21 valores float32, 84 bytes lógicos por muestra antes de la
compresión. El cambio del tamaño de archivo incluye codificación y metadatos,
no mide por sí solo la RAM que necesita el entrenamiento.

La materialización final tardó 7,44 segundos dentro del proceso, incluida la
carga de codificadores, y registró 1.948,09 MiB de RSS máximo. La ejecución
coincidió con validaciones locales y una auditoría de lectura del corpus.
No es una medida aislada ni un ensayo de aceleración. La preparación procesa
un activo cada vez y escribe bloques de ocho muestras en esta comprobación.

## Entrenamientos ejecutados

Las tres referencias usan las mismas 50 etiquetas de entrenamiento hasta 2022
y las mismas 15 de validación de 2023. La reserva final no se ha usado para
ajustar ni seleccionar. Estas ejecuciones comprueban la integración de la
representación, no constituyen una comparación concluyente del corpus.

| Referencia | Ajuste medido | MAE de validación | MSE de validación |
| --- | ---: | ---: | ---: |
| Ridge, alpha 1 | 0,187 s | 0,017878 | 0,000491 |
| HistGradientBoosting, 30 iteraciones | 0,861 s | 0,011771 | 0,000197 |
| GRU, 3 épocas | 0,263 s | 0,066048 | 0,005936 |
| Retorno cero | Sin ajuste | 0,009620 | 0,000160 |

Los tiempos de ajuste no incluyen todo el arranque, preparación, validación
y guardado. Cada registro delimita su medición:
[Ridge](../resources/company-ridge-probe.json),
[boosting](../resources/company-boosting-probe.json) y
[GRU](../resources/company-gru-probe.json).
Ridge utiliza `cuda:0`. HistGradientBoosting se ejecuta expresamente en CPU.
La GRU utiliza `cuda:0`, float32, lotes de 16 y 52.609 parámetros.
Ridge y boosting se ejecutaron en paralelo. La GRU se ejecutó después.
La auditoría contable de un proceso permaneció activa durante las mediciones.

Se ha comprobado la igualdad de predicciones tras restaurar los tres modelos.
En la GRU, repetir desde el checkpoint de la primera época reproduce exactamente
los pesos finales. Esta prueba no sustituye la continuación operativa de una
campaña interrumpida, que permanece pendiente en la tarea de recuperación.

Ningún modelo supera el MAE de retorno cero en esta muestra. No se atribuye a
los ratios una mejora causal frente a los ensayos anteriores. En particular,
ampliar la entrada de la GRU también cambia sus parámetros y su inicialización.
Todavía faltan población más amplia, varias semillas y presupuestos comparables.
No se ha entrenado MARS-TITAN ni añadido aprendizaje por refuerzo.

## Comprobaciones de la implementación

La suite completa ejecutó 496 pruebas sin omisiones, con CUDA, los codificadores
reales y clang-tidy configurado. Las cuatro mutaciones dirigidas se detectaron:
aceptar denominador cero, usar flujos como saldos, ignorar la presentación y
fechar un ratio con su componente más antiguo.

La revisión independiente detectó una diferencia de retención entre una
presentación posterior con solo caja y otra con solo activo total. La regresión
se reprodujo y se corrigió registrando ambos grupos antes de seleccionar
componentes. También se prueban conflictos tardíos, unidades incompatibles,
desbordamientos, escritura incompleta y cachés dañadas.

Coverage.py 7.16.1 cubre las 95 sentencias del módulo de factores y 52 de sus
54 ramas. Radon 6.0.1 mide complejidades de 8, 21, 18 y 5 en sus cuatro
funciones principales. El [registro de calidad](company-factor-quality.json)
incluye CRAP calculado con cobertura de sentencias, las mutaciones seleccionadas
y los tiempos de proceso completo medidos por GNU time. Estos resultados no
demuestran ausencia de errores ni validan la capacidad predictiva.

La [definición del cálculo](../../docs/engineering/company-factors.md) describe
las fórmulas y sus límites. La auditoría de todos los conceptos del corpus y la
ampliación de la intersección editorial siguen abiertas en #94 y #66.
