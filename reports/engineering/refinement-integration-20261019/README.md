# Comprobación de integración de las rutas de refinamiento

La integración `073f6e39ff9cffb362915375d888d8c96c1b3c27` combina las rutas de
refinamiento con las correcciones numéricas revisadas de `develop`. Pasaron
659 casos, incluidos 59 CUDA, sin fallos ni omisiones. El proceso duró
896,026 segundos. No es una repetición de la batería completa del repositorio.

Se comprobaron rutas, gradientes, lectores heredados, atención al congelar
parámetros, políticas por umbral y hash, contratos de selección, recuperación,
cuantiles, preparación de comparaciones y latencia. La prueba de campaña usa
fixtures con siete ventanas de ajuste, dos casos de búsqueda por ventana y
doce traslados. Se conservaron sus treinta épocas. Esa ejecución no consume
el corpus real ni abre el test final.

La comprobación CUDA utilizó `cuda:0` en una RTX 4070 Laptop, con PyTorch
2.14.0+cu130 y CUDA 13.0. La lista exacta de casos CUDA, archivos, huellas y
horas está en [verification.json](verification.json). PyTorch emitió un aviso
sobre el carácter experimental de su diagnóstico de sincronización. No se
silenció. Los tiempos de esta batería instrumentada no son medidas de latencia
ni de capacidad del predictor.

La cobertura con ramas se obtuvo en esta misma invocación. El resumen de las
fuentes indicadas en [quality.json](quality.json) contiene 3.434 de 3.777
sentencias y 1.028 de 1.292 ramas cubiertas, con 282 funciones medidas.
Se usan coverage.py 7.16.1 y Radon 6.0.1. CRAP sigue la convención
`C² × (1 − cobertura de sentencias)³ + C` del resumidor del proyecto.

El máximo CRAP de ese alcance es 145,703 en `FinancialConfig.__post_init__`,
con 28 de 47 sentencias recorridas. La regresión de integración no cubre todas
las combinaciones de ese constructor. Este valor identifica una limitación
de la cobertura recogida, no demuestra un fallo ni permite afirmar cobertura
global del 100 %. Se conservan las treinta funciones con mayor CRAP y los
denominadores del conjunto medido.

Las tres mutaciones dirigidas fueron detectadas. Eliminaban la normalización
de identidad al emparejar controles activos e inactivos, cambiaban la identidad
histórica de C desactivado o reactivaban los hilos del cálculo JVP. Todas produjeron fallos
de comportamiento, sin errores de colección. Al restaurar los archivos,
pasaron los quince casos CPU elegidos. Los ocho casos CUDA de esa selección se
excluyeron de la ronda de mutaciones CPU y sí forman parte de la regresión
CUDA completa de esta integración. [mutations.json](mutations.json) conserva
las sustituciones, pruebas y hashes. Son tres mutaciones concretas, no una
campaña de mutación de todo el repositorio.

Pasaron `uv run --no-sync ruff check .`,
`uv run --no-sync ruff format --check .` y la comprobación de repositorio.
Las autorizaciones de fixtures se cerraron al acabar y la protección global
de aprendizaje mantuvo su huella.

Los [resultados científicos de la ronda](../../research/refinement-development-20261019/completed-v1/README.md)
permanecen vinculados al runtime `6b037a7262d4c2d97105b194a137f8b2f5767b01`.
Esta comprobación de integración no recalcula ni sustituye esos resultados.
La política sigue siendo experimental y está desactivada por defecto.
