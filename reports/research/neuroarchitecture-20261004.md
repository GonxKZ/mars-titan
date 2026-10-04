# Comprobación de la revisión de memoria y contexto

Fecha: 4 de octubre de 2026. Alcance: [revisión de arquitectura](../../docs/research/neuroarchitecture-review.md), [apéndice matemático](../../docs/research/memory-mathematics.md), bibliografía y programa de comprobación algebraica. No se implementó ni entrenó MARS-TITAN y no se modificaron datos, configuraciones o fuentes de la campaña activa.

El catálogo incorpora 27 referencias y suma 150 identificadores únicos, coincidentes con BibTeX. Las fichas distinguen publicación, versión leída, pasajes consultados y acceso pendiente. Se comprobaron las portadas de los PDF históricos de ATLAS y MIRAS y sus hashes frente al manifiesto. Ambos corresponden a v1. El manifiesto y los archivos originales permanecen intactos.

El verificador del repositorio y las 13 pruebas de biblioteca pasaron. Ruff comprobó el código y el formato. Los enlaces locales, los catálogos y las entradas bibliográficas se validaron con las herramientas existentes.

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python scripts/check_repository.py
uv run --no-sync python -m pytest tests/tooling/test_library.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  reports/research/check_memory_algebra.py
```

Las comprobaciones algebraicas usan NumPy 2.5.3 en CPU y FP64. Incluyen 360 casos de la regla escalar, 162 sistemas proximales, un operador con amplificación transitoria, un contraejemplo de información condicional y dos ejemplos sobre ruido y pérdidas recortadas. El [registro numérico](memory-algebra-20261004.json) identifica el programa mediante SHA-256. Son comprobaciones de ecuaciones y condiciones, no resultados de aprendizaje ni mediciones de rendimiento.

La [medición de cobertura y complejidad](memory-algebra-quality-20261004.json) utiliza coverage.py 7.16.2 y Lizard 1.24.0. Se ejecutaron las salidas a consola y a archivo. Se cubrieron 110 de 111 líneas ejecutables y 14 de 16 ramas. CRAP se calcula con la cobertura de líneas de cada función y la fórmula CCN² × (1 − cobertura)³ + CCN. El valor máximo es 6. La línea no recorrida es el lanzamiento de error de la función de comprobación en una ejecución válida. Las ramas pendientes incluyen ese fallo y la importación del programa sin ejecutarlo como comando.

Tres mutaciones dirigidas fueron detectadas por las aserciones:

- Eliminar el olvido del operador escalar.
- Transponer el término de borrado por canal.
- Dividir la suma de Gram entre el número de microlotes, alterando los pesos globales.

Las copias mutadas y sus registros permanecen fuera del código publicado. La mutación no fue exhaustiva y la cobertura no prueba un teorema. La lectura crítica corrigió la distinción entre condición suficiente y necesaria de contracción, la separación entre experiencia y habilidad y la localización de la no expansión proximal en §2.3 de la copia de los autores.

No se ejecutaron rutas CUDA porque el cambio no modifica ni añade una ruta CUDA. Las comprobaciones algebraicas en CPU no sustituyen las pruebas de un futuro candidato nativo. Quedan pendientes su implementación autorizada, paridad, recuperación, perfilado y evaluación predictiva. El test final permanece cerrado.
