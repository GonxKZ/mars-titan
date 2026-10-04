# Verificación de atención, repetición y presupuesto de señales

Fecha: 4 de octubre de 2026. Alcance: [atención y replay](../../docs/research/attention-replay-review.md), [eventos y señales reducidas](../../docs/research/event-signal-comparison.md), [matemáticas](../../docs/research/attention-replay-mathematics.md), bibliografía y programa de comprobación. Son diseños separados del candidato y ejemplos algebraicos. No se implementó ni entrenó MARS-TITAN ni se modificó la campaña activa.

Se incorporaron 40 referencias, con un total de 190 identificadores concordantes entre JSON y BibTeX. No se encontraron duplicados de título o DOI entre esas incorporaciones y el catálogo previo. Las fichas conservan las versiones leídas, los pasajes y las discrepancias editoriales. La entrada de Abbes permanece anclada al preprint consultado. Rahnev y Butlin mantienen su lectura parcial explícita. El manifiesto histórico de descargas no se ha reescrito.

Las comprobaciones locales de Ruff, formato y repositorio pasaron, junto con las 13 pruebas de biblioteca. Se verificaron enlaces locales, catálogos y dependencias. El análisis del catálogo macro confirmó 70 entradas `raw` y 70 `derived`. Eso no demuestra que basten los últimos 70 valores para reconstruir toda su historia.

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python scripts/check_repository.py
uv run --no-sync python -m pytest tests/tooling/test_library.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  reports/research/check_attention_replay_algebra.py
```

El [programa](check_attention_replay_algebra.py) utiliza NumPy 2.5.3 en CPU y FP64. El [registro](attention-replay-algebra-20261004.json) contiene fecha, versión, semillas, tolerancias y SHA-256 del programa. Comprueba 27 casos de muestreo por importancia, 81 recortes de pesos, tres casos con probabilidad cero fuera del soporte y seis repeticiones de una etiqueta contaminada. Añade ejemplos de dependencia entre copias, señales no observadas, destilación, actualización bayesiana, calibración y curvatura.

El error máximo de la identidad de esperanza y de la fórmula del sesgo por recorte fue 2,22 × 10⁻¹⁶. El ejemplo con siete observaciones copiadas cuatro veces conserva varianza de la media 0,357142857, frente al 0,089285714 que resultaría de tratarlas incorrectamente como independientes. El ejemplo bayesiano pasa de 0,75 a 0,995901639 al contar cinco copias como evidencia independiente. Son controles con supuestos declarados, no estimaciones de esos efectos en el corpus.

La [medición de calidad del comprobador](attention-replay-quality-20261004.json) usa coverage.py 7.16.2 y Lizard 1.24.0. Se ejecutaron las salidas a consola y a archivo. La cobertura fue de 126/127 líneas ejecutables y 16/18 ramas. CRAP utiliza la fracción de líneas de cada función, con CCN² × (1 − cobertura)³ + CCN. Su máximo fue 3. La línea pendiente es el lanzamiento de error de la función de comprobación durante una ejecución válida. Las ramas pendientes incluyen ese fallo y la importación sin ejecutar el comando.

Cuatro mutaciones dirigidas fueron detectadas: retirar la corrección por importancia, dividir por probabilidad cero, tratar las copias como independientes y omitir el término de curvatura. Se detectaron mediante aserciones o error de coma flotante, según el caso. No fue una campaña exhaustiva de mutación.

La revisión matemática precisó el soporte de los cocientes, la tolerancia del segundo momento y la condición de transiciones homogéneas del HMM. La revisión del protocolo distinguió censura al final del seguimiento y huecos que dejan incompleta la historia de eventos. Las correcciones están incorporadas.

No se han medido rendimiento, RAM, VRAM o energía de los diseños propuestos. Las pruebas algebraicas no sustituyen evaluación predictiva, paridad de un futuro backend ni comprobaciones CUDA. No se añadieron rutas CUDA en esta revisión. El test final permanece cerrado.
