# Verificación de la preparación y los ensayos

El 19 de septiembre de 2026 pasaron 317 pruebas locales, sin omisiones, en
24,89 segundos. Ruff y su comprobación de formato también terminaron sin errores.
El [registro estructurado](quality.json) conserva herramientas, cobertura,
complejidad, hashes de código y las variantes de fallo comprobadas.

## Comportamiento comprobado

La suite cubre inventario reanudable, integridad de originales, calendarios,
fechas de publicación, conflictos contables, fórmulas macro, unidades históricas,
gráficos causales, corrupción de caché, particiones y recuperación de cursores.
Las salidas rechazan rutas que alcancen originales o preparados, incluidos enlaces
simbólicos entre los dos mercados. Un calendario distinto invalida la reutilización
antes de cargar codificadores o abrir la caché.

Una integración con CUDA real prepara 22 activos sintéticos y entrena cuatro con
MLP y GRU durante dos épocas. Contrasta un residual calculable a mano, lotes
parciales, fronteras temporales y actualización de las cuatro modalidades y macro.
Una fila futura deliberadamente inválida detecta su consumo accidental. La
recuperación desde la primera época reproduce exactamente los pesos y el estado
aleatorio comprobado de la segunda. Ninguna de estas cifras sintéticas representa
rendimiento predictivo en mercado.

Las [ejecuciones con datos reales](campaign-budget.md) son independientes de esa
prueba y de la instrumentación de cobertura. Se verificaron las 115 huellas de
entrada de cada informe. El piloto se reconstruyó con la misma huella de manifiesto,
`94370898e8084da97e6624c49eafd685cd39e3f9c70f1f84380f79720f528784`.

## Cobertura y complejidad

Se utilizó coverage.py 7.16.1 con seguimiento de ramas sobre `src/mars_titan`.
La suite cubre 1.819 de 2.217 sentencias, un 82,05 %, y 570 de 794 salidas de
rama, un 71,79 %. La medida combinada de la herramienta es 79,34 %. No se mezclan
estos tres porcentajes ni se añaden las ejecuciones externas a su denominador.

Para CRAP se utilizó la complejidad ciclomática de cada función o método según
Radon 6.0.1 y la cobertura de sentencias de esa función según coverage.py:

\[
\operatorname{CRAP}=C^2(1-c)^3+C.
\]

Se analizaron 116 funciones y métodos, sin puntuar agregados de clases ni cierres
anidados por separado. La orquestación `train_budget_grid` tiene C = 26, cobertura
de sentencias del 96,83 % y CRAP = 26,02. Su integración real permite comprobar
algo más útil que la mera existencia de una función o una línea de configuración.

La selección completa del piloto, la selección técnica y varias envolturas de
adquisición o perfilado siguen con cobertura automática baja o nula. Sus valores
CRAP altos aparecen en el JSON, aunque algunas se hayan ejecutado con datos reales.
Son prioridades para ampliar las regresiones. Las ejecuciones manuales no se
presentan como cobertura automatizada ni una cifra de cobertura como garantía.

## Mutación dirigida

Se comprobaron 15 variantes en procesos aislados, alterando el código únicamente
en memoria. Las 15 provocaron fallos de las pruebas correspondientes. Once prueban
calendarios, orden de validación y protección de rutas. Cuatro comprueban la
actualización del optimizador, el cursor de recuperación, la exclusión del test
final y la portabilidad de la procedencia. Dos de las once primeras desactivan varias guardas redundantes
a la vez y se identifican como variantes compuestas. No es una puntuación exhaustiva
de todas las mutaciones posibles del proyecto.

## Repetición

```bash
uv sync --locked --extra cuda --extra encoders
uv run --locked --extra cuda --extra encoders ruff check .
uv run --locked --extra cuda --extra encoders ruff format --check .
uv run --locked --extra cuda --extra encoders pytest
uv run --locked --extra cuda --extra encoders python scripts/check_repository.py
uv run --extra cuda --extra encoders --with coverage==7.16.1 \
  --with radon==6.0.1 python -m coverage run --branch --source=src/mars_titan -m pytest -q
uv run --with coverage==7.16.1 python -m coverage json -o tmp/coverage.json
uv run --with radon==6.0.1 radon cc src/mars_titan -j
```

La integración CUDA se omite de forma explícita si no hay dispositivo compatible.
En el equipo medido se ejecutó. No se ha comprobado una interrupción eléctrica real,
una campaña sostenida de 24 horas ni la recuperación de una memoria adaptativa que
todavía no está implementada. El universo completo tampoco está convertido a Parquet.
