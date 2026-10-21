# Núcleo del selector de mecanismos y objetivos separados

`b19d064a0` añade el cálculo de utilidades sobre las 36 combinaciones de contexto, ruta y calibración de [#701](https://github.com/GonxKZ/mars-titan/issues/701). Sus coeficientes están dados, no ajustados. Este incremento no conecta todavía el selector a `FinancialSession`, no extrae observaciones del corpus y no cambia la ruta por defecto. No constituye un controlador entrenado ni una mejora predictiva medida.

## Decisión y objetivos

[`MechanismPolicy`](../../../src/mars_titan/models/titans/mechanism_policy.py) recibe uno o dos vectores de 21 rasgos y una máscara de admisibilidad por cohorte de mercado. El esquema declara una decisión anterior a MAC. Sus nombres reservan presencia y magnitud de las cinco fuentes, edad de las entradas, tamaño del banco, contexto anterior y calibración. La extracción y la comprobación temporal de esos valores siguen pendientes. Recibir un tensor con esa forma no demuestra que proceda del prefijo permitido. Tampoco calcular una máscara acredita que se hayan aplicado las precondiciones del catálogo. Esa integración debe comprobarse por separado.

El núcleo mantiene doce regresiones de ganancia puntual, una por contexto y ruta, y 36 de ganancia de intervalos. Las tres calibraciones de una misma combinación comparten exactamente su regresión puntual. PT2 conserva la mediana, por lo que no puede recibir aquí una ganancia puntual distinta solo por activar la calibración.

Para el vector estandarizado z y la acción a, se calcula:

$$
s_a = \widehat G_{\mathrm{MAE},\,g(a)}(z)
      + \lambda_I\widehat G_{\mathrm{IS},\,a}(z)
      - c_a - \delta.
$$

La referencia `off/k1/none` mantiene puntuación cero. Sus coeficientes incrementales y coste deben ser cero. El peso de intervalos λ_I, la mejora mínima δ y los costes son valores explícitos. Los costes ya deben estar expresados en las unidades de la utilidad declarada por el protocolo. El módulo no convierte milisegundos ni elige retrospectivamente esas escalas.

Se excluyen las acciones no admisibles antes de escoger la puntuación máxima. Los empates siguen el orden del catálogo y favorecen la referencia cuando empata a cero. La referencia debe ser admisible. Una entrada inválida o un desbordamiento se rechazan. Fuera de los rangos marginales declarados se devuelve la referencia, sin evaluar la regresión sobre valores extremos no utilizados. Estar dentro de esos rangos no demuestra soporte conjunto ni es un detector calibrado de distribución desconocida.

La elección devuelve puntuaciones, las dos ganancias previstas, admisibilidad y estado del soporte. Son estimaciones de utilidad, no pérdidas observadas de acciones descartadas. La inferencia no conserva autograd, no consume RNG y no modifica coeficientes. Los checkpoints comprueban identidad, campos, dimensiones, precisión, finitud, escala, soporte y referencia antes de copiar los buffers.

[`mechanism_losses`](../../../src/mars_titan/models/titans/mechanism_losses.py) separa el error absoluto puntual y el interval score medio de los intervalos centrales del 80 % y 95 %. Devuelve también anchura y cobertura con extremos incluidos. No promedia activos como si fueran sesiones. Conserva ganancias negativas respecto a la referencia y rechaza alternativas incompletas, cuantiles cruzados o medianas distintas entre calibraciones del mismo contexto y ruta.

El interval score usa [Gneiting y Raftery (2007), sección 6.2, ecuación 43](https://doi.org/10.1198/016214506000001437). Se inspeccionó esa sección de la [copia publicada por el autor](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). Suma anchura y penalizaciones por exceder los extremos, con factor 2/α. Esa propiedad del objetivo de intervalos no garantiza cobertura en una secuencia seleccionada ni convierte la regresión de utilidad en una predicción financiera correcta. No es un resultado nuevo del proyecto.

La función de pérdidas recibe tensores de etiquetas. No acredita por sí sola su madurez ni la procedencia de las alternativas. Su futuro colector debe conservar las predicciones calculadas antes de la etiqueta desde snapshots identificados y distinguir una alternativa no observada de pérdida cero. El ajuste, sus cortes y la validación predictiva siguen pendientes.

## Evidencia técnica

La [selección de pruebas](verification.json) pasó 90 casos, nueve marcados CUDA, sin fallos ni omisiones. Incluye 51 casos nuevos, ocho CUDA, y los controles anteriores del catálogo y de las seis rutas. No es toda la suite del repositorio. Los tests nuevos empezaron con fallos por módulos ausentes. Los [artefactos externos](external-artifacts.json) conservan esos XML y logs junto con las comprobaciones posteriores.

Las pruebas usan coeficientes y tensores manuales. Contrastan las puntuaciones con cálculos directos, costes y mejora mínima, acciones excluidas, ganancias negativas, orden y partición, aislamiento del RNG y de autograd, soporte marginal, desbordamiento, FP32/FP64, CPU/CUDA, rechazo de autocast y recuperación de la siguiente elección desde un checkpoint CPU en CUDA. Un ejemplo mantiene la misma mediana y obtiene interval scores distintos al cambiar las colas. Son comprobaciones aritméticas, no resultados sobre activos reales.

Se detectaron las [doce mutaciones dirigidas](mutations.json). Alteraron el reparto puntual entre calibraciones, el peso de intervalos, costes, mejora mínima, admisibilidad, soporte, autograd, identidad, igualdad de medianas, penalización de excedencias, ganancias negativas y cobertura en los extremos. Las referencias inicial y restaurada pasaron 41 casos CPU cada una, sin omisiones. Después se añadieron comprobaciones de precisión y carga CUDA. Las definiciones anteriores permanecen iguales en el AST. El JSON conserva las dos revisiones de las pruebas. No se afirma una campaña exhaustiva de todas las mutaciones posibles.

La [cobertura y CRAP](quality.json), con coverage.py 7.16.1 y Radon 6.0.1, registran 122/122 sentencias y 28/28 ramas en las ocho funciones de los dos módulos nuevos. El CRAP utiliza cobertura de sentencias y su máximo es 14 en `MechanismPolicy.forward`. Estas cifras describen esta selección y no prueban la causalidad de una observación que todavía no se extrae ni la ausencia de todos los errores posibles.

Pasaron Ruff, el formato de 1.719 archivos y la revisión inicial del repositorio. No se midieron todavía latencia, caudal, RAM, VRAM ni coste de decisiones financieras de este selector. El tamaño de sus matrices no sustituye esas medidas. Faltan el enlace de las observaciones, las trazas de selección, la recuperación integrada y sus controles, además de la colección y el ajuste protegidos. No se ha ejecutado ningún ajuste de regresión, paso de optimizador, entrenamiento nuevo, corpus real ni test final.

La selección técnica se reproduce en el entorno del proyecto con CUDA disponible:

```bash
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_REQUIRE_CUDA=1 uv run --no-sync pytest -q tests/models/titans/test_mechanism_policy.py tests/models/titans/test_mechanism_losses.py tests/models/titans/test_mechanism_catalog.py tests/models/titans/test_refinement_policy.py
```
