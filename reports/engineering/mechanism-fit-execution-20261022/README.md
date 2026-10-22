# Ejecución numérica del ajuste de mecanismos

El 22 de octubre de 2026 se ejecutó el ajuste ya implementado de `MechanismPolicy` sobre sesiones financieras nativas de prueba. La pasada `final5` aprobó 13 casos en CUDA, sin omisiones, en 62,85 segundos. El proceso supervisado duró 70,80 segundos. La batería posterior de 83 casos completó sus comprobaciones en una sola pasada sin fallos, errores ni omisiones, en 39,03 segundos, con 41,47 segundos de proceso. Esta última pasada no instrumentó cobertura. El cambio añade una referencia independiente y pruebas de ejecución. No modifica el algoritmo de producción ni acredita una mejora predictiva.

Las fuentes son cuatro flujos sintéticos del mercado US, con precios, gráficos y distintas ausencias de noticias y fundamentales. Sus identificadores parten de la misma serie y calendario de prueba, con perturbaciones posteriores. No son cuatro activos financieros independientes ni amplían el corpus real. El contexto macro está ausente. Las etiquetas se asignan manualmente y no se estiman retornos residuales. Se reutilizan el registro nativo C++20, Titans-MAC, el lector regional y la admisión M3. Los parámetros del predictor permanecen congelados. La actualización interna de la memoria neuronal y el ajuste ridge de los buffers de la política son operaciones distintas. Un gancho rechaza cualquier paso de optimizador de PyTorch.

El protocolo técnico utiliza ridge 0,01, 21 rasgos, 12 objetivos puntuales y 36 objetivos de intervalos, con mínimo de dos cohortes. Este mínimo permite comprobar instancias pequeñas. El mínimo de producción sigue siendo 32 cohortes. Ninguno constituye una garantía de generalización. La prueba no selecciona al padre ni valida su calidad financiera.

El contraste independiente calcula las pérdidas desde los cuantiles sellados de las alternativas y etiquetas manuales, obtiene momentos con NumPy y resuelve ecuaciones normales en FP64. Las alternativas se calculan antes de recibir sus etiquetas, pero no todas son emisiones de la trayectoria financiera. Producción resuelve un sistema aumentado con `torch.linalg.lstsq(..., driver="gels")` en `cuda:0`. La comparación comprueba coeficientes, normalización, soporte y máscaras con tolerancias absoluta de 1e-12 y relativa de 3e-11. Las dos regresiones de referencia conservan coeficientes cero y el intercepto también está penalizado.

Se comprueban estos recorridos:

- Fuentes cerradas y ventanas nativas con los mismos buffers cuando contienen la misma evidencia.
- Pesos iguales por cohorte aunque sus números de etiquetas maduras sean 3, 4 y 4.
- Objetivos completamente ausentes y objetivos observados solo en la primera y tercera cohorte. Los ceros de almacenamiento no se convierten en observaciones.
- Una etiqueta disponible antes del corte pero recibida después. El ajuste temprano utiliza el prefijo anterior a su recepción.
- Escritura de pesos y recibo, carga de pesos en un consumidor, enlace de identidad y conservación de los generadores aleatorios CPU y CUDA.
- Rechazo explícito de CPU, autocast y condicionamiento por encima del límite declarado.

El equipo utilizado tiene una RTX 4070 Laptop de 8.188 MiB, compartida con otra carga. Se mantiene FP64 y no se utiliza TensorRT ni se relajan tolerancias para acelerar esta comprobación. El registro de sesiones utiliza el módulo nativo existente y el álgebra lineal CUDA de PyTorch. No se ha compilado un kernel nuevo ni medido una aceleración con esta batería. Sus tiempos incluyen preparación de fuentes y cobertura. No son latencias de inferencia ni medidas del coste de una campaña.

## Límite observado del arranque

La política ajustada con el prefijo temprano solo observa `previous_context_present=1`. Una fase nueva, calentada sin emitir predicciones, empieza con ese rasgo en cero. El selector detecta el estado fuera de su soporte y pide `off/k1/none`. Sus predicciones coinciden con las de esa referencia fija.

Esta acción tampoco crea el contexto PT4 que falta. Por la regla de actualización, el retorno puede mantenerse en decisiones posteriores. La prueba ejecuta la primera decisión de la nueva fase. La permanencia posterior se deduce del contrato de actualización y no se presenta como una trayectoria larga ejecutada. El rango de soporte se conserva. Falta definir y contrastar una política de arranque y evaluar después una trayectoria autónoma dentro del soporte.

## Alcance de la evidencia

El [registro de ejecución](verification.json) conserva los intentos anteriores, sus fallos y las huellas de las fuentes. Incluyen expectativas incorrectas sobre censura, el contenedor de contexto vacío, residuos cuando no se resuelve ningún objetivo y una fecha ausente del fixture. La revisión posterior también corrigió la preparación del fastpath y comparó el recibo JSON por su forma canónica, porque la serialización convierte tuplas en listas. La pasada `final5` comprueba esas correcciones. No se cambió producción para hacerlas pasar.

La selección ampliada anterior aprobó 70 casos de contratos y registró 13 errores de preparación del fixture nuevo. Otra pasada aprobó 82 casos y detectó la comparación incorrecta del recibo. Un intento conjunto posterior alcanzó su límite de 240 segundos y no completó el informe de pytest. Esos intentos no se presentan como una batería conjunta aprobada. Tras corregir las causas, se completan los 13 casos con cobertura y la pasada conjunta de 83 sin esa instrumentación. La diferencia de tiempos no se atribuye a una optimización del modelo.

La [mutación dirigida](mutations.json) detecta los 12 cambios declarados. Cubren el signo de las ganancias, la ponderación por filas, la escala, la penalización ridge y su intercepto, las cohortes inmaduras, el enlace del recibo, las máscaras y la renormalización de objetivos parcialmente observados. Cada detección corresponde a una aserción del caso seleccionado, sin errores de preparación ni omisiones. El control inicial pasa. La ronda alcanzó su presupuesto de 900 segundos antes de completar el control restaurado y conserva el estado `incomplete`. Solo ese control se ejecutó después con un límite adicional de 240 segundos y pasó en 34,26 segundos. No se repitieron los doce mutantes ni se reescribió el resultado anterior. La mutación usa la copia congelada anterior a las últimas comprobaciones adicionales del recibo y la limpieza del fixture. Sus huellas figuran por separado. Es un conjunto finito dirigido, no una puntuación global de mutación.

La [cobertura y CRAP](quality.json) corresponden solo a la pasada final de los 13 casos. Se calculan con coverage.py 7.16.1, registro de ramas y Radon 6.0.1 mediante `scripts/summarize_python_quality.py`. La convención usa cobertura de sentencias por región de función para CRAP y publica ramas aparte.

| Módulo | Sentencias ejecutadas | Ramas ejecutadas |
| --- | ---: | ---: |
| `mechanism_supervision.py` | 108/108 | 17/18 |
| `mechanism_fit_window.py` | 66/80 | 9/14 |
| `mechanism_policy_fit.py` | 153/154 | 23/24 |

`fit_policy` tiene complejidad 21 y CRAP 21,0014 en esta pasada. `audit_fit_windows` no se ejecuta en esos 13 casos y conserva CRAP 90 en este diagnóstico. Sus pruebas anteriores son otro alcance. Las cifras no acreditan una cobertura total ni demuestran ausencia de errores.

El grupo numérico se ejecutó con este comando, dentro de un proceso que autorizaba únicamente los fixtures técnicos y conservaba cerrada la protección de campaña:

```bash
MARS_TITAN_REQUIRE_CUDA=1 uv run --no-sync coverage run --branch \
  --source=src/mars_titan/training,src/mars_titan/memory \
  -m pytest -q tests/memory/test_mechanism_policy_fit_execution.py -o addopts=
```

Se requiere el enlace episódico existente en `MARS_TITAN_EPISODIC_NATIVE`. El lanzador real añadió destinos privados para XML y archivos de prueba y un límite de 240 segundos. Los pesos, registros nativos y permisos permanecen fuera del repositorio. El JSON publicado conserva sus huellas. Ejecutar el comando bajo la protección general no concede una excepción para aprender.

## Sonda de resolución conjunta en CUDA

El [microbenchmark](solver-probe.json) compara resolver cada objetivo por separado con agrupar segundos miembros que tienen exactamente la misma máscara y los mismos pesos. Ambas opciones utilizan FP64, la misma penalización del intercepto y `gels`. No hay un kernel propio. El [código ejecutado](../../../benchmarks/grouped_ridge_cuda.py) conserva la referencia con bibliotecas CUDA de PyTorch.

Se midieron 3, 32, 512 y 4.096 filas, con disponibilidad completa y máscaras mezcladas, tres calentamientos y siete repeticiones. Una segunda pasada invierte el orden de las variantes dentro de cada condición. Las dieciséis comparaciones cumplen las tolerancias declaradas. La relación entre las medianas del núcleo medido va de 8,66 a 33,45. Es un rango de observaciones, no un intervalo de confianza ni la aceleración del ajuste completo.

Con 4.096 filas y todos los objetivos observados, las medianas pasan de 195,07 a 9,19 milisegundos y de 189,70 a 8,30 milisegundos en las dos pasadas. Los grupos pasan de 46 a uno. Cada grupo ejecuta una SVD y una resolución GELS, sin tratar esas llamadas como launches físicos medidos. El pico adicional del allocator aumenta de 12.860.928 a 16.668.160 bytes en ese caso, unos 3,63 MiB más. No hay un ahorro universal de memoria.

Las matrices y máscaras CPU y CUDA están preparadas antes del tramo medido. Este incluye normalización, formación de grupos, SVD, resolución, diagnósticos y sincronización. No incluye lectura de fuentes, creación de sesiones o publicación de artefactos. La GPU está compartida. Siete muestras no permiten interpretar p95 como una estimación estable de cola. La preferencia cuSOLVER de PyTorch utilizada por la sonda está marcada como experimental por la biblioteca.

Para repetir las dos pasadas sin escribir sus salidas en el repositorio:

```bash
ridge_probe_dir="$(mktemp -d)"
cp benchmarks/grouped_ridge_cuda.py "$ridge_probe_dir/"
uv run --no-sync python "$ridge_probe_dir/grouped_ridge_cuda.py" --run --label normal
uv run --no-sync python "$ridge_probe_dir/grouped_ridge_cuda.py" --run --label reverse --reverse
```

Cada proceso tiene un límite de 90 segundos y acota el allocator, sin limitar toda la memoria CUDA externa. La agrupación no está integrada en `fit_policy`. Falta comprobar su efecto en ese recorrido completo antes de adoptarla. Esta sonda no modifica pesos de un predictor ni mide su error financiero.

La [especificación del ajuste](../../../docs/engineering/mechanism-policy-fitting.md) mantiene sus límites de procedencia y los [informes anteriores](../mechanism-fit-window-20261021/README.md) conservan la evidencia de transporte sin solver. Un hash de recibo permite contrastar integridad. No autentica por sí solo una política seleccionada científicamente.

Quedan pendientes el contraste numérico independiente entre mercados con pesos distintos, la selección temporal del controlador y su comparación con controles fijos, reglas simples, mezclas constantes y residual cero. No se ha ejecutado una comparación financiera en esta batería. La campaña completa y el test final siguen cerrados. La [issue #701](https://github.com/GonxKZ/mars-titan/issues/701) conserva esos criterios abiertos.
