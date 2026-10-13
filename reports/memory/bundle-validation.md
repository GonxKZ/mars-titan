# Compatibilidad de versiones en la sesión financiera

Autor: Gonzalo García Lama. Comprobación: 10 de octubre de 2026.

Este informe recoge la evidencia de MT-059 (#61). Una sesión financiera reúne una edición del corpus, su índice de observaciones, el predictor congelado, el codec episódico, el verificador de prefijos y el ejecutor nativo. La prueba de extremo a extremo [`test_bundle_compatibility.py`](../../tests/memory/test_bundle_compatibility.py) mezcla a propósito piezas de versiones distintas y comprueba que cada mezcla se rechaza antes de escribir o sin cambiar lo ya confirmado.

## Cómo se prepara

Cada caso construye ediciones reales del corpus histórico con máscaras a partir del mismo fixture: dos ediciones A y B con contenido equivalente en carpetas distintas y una edición R igual que A pero con otra revisión de sus codificadores. Las tres tienen las mismas dimensiones (precios 5, noticias 2, gráficos 1, fundamentales 3 y macro 3), así que un rechazo solo puede venir de las identidades. El predictor es Titans-MAC en FP64, en evaluación y sin gradientes, y el ejecutor es el enlace nativo del candidato en CPU. No hay pasos de optimizador.

## Casos

| Mezcla | Resultado | Dónde se detiene |
| --- | --- | --- |
| Edición A completa, recorrida hasta el cierre de la fase y reanudada | Aceptada, con la misma instantánea | |
| Predictor de A con el codec de B | Rechazada sin crear la carpeta | `TitansBinding.check`: el codec no comparte la especificación del predictor |
| Predictor y codec de A con el verificador de prefijos de B | Rechazada sin crear la carpeta | `FinancialSession`: la fuente del prefijo no es la de la especificación |
| Lectura episódica ajustada para el codec de B | Rechazada sin crear la carpeta | `TitansBinding.check`: la lectura no lleva la huella del codec |
| Supervisión de A con la edición materializada de B | Rechazada | `PrefixTargetVerifier`: el calendario no procede de la edición materializada original |
| Índice de observaciones de B en una sesión de A | Rechazada sin cambiar archivos | `run_observation_source`: la sesión no está vinculada a ese índice |
| Lote de B validado con su especificación y entregado a mano a la sesión de A | Rechazado sin cambiar archivos | El codec episódico. El predictor lo rechaza también por sí solo |
| Reanudar la sesión de A con todas las piezas de B | Rechazada sin cambiar archivos | Ejecutor nativo: la ejecución pertenece a otro origen, vista o modelo |
| Reanudar A con otra receta del predictor (`hidden_size=64` o las proyecciones del artículo de #474) | Rechazada sin cambiar archivos | Ejecutor nativo, por la identidad del modelo en el contrato |
| Reanudar A con otra regla de admisión (M1 en lugar de M0) | Rechazada sin cambiar archivos | Ejecutor nativo, por el contrato de la sesión |
| Cohorte con un activo codificado con otra representación | Rechazada al abrir el corpus | `cohort_identity`: las representaciones no conservan una identidad común |
| Observaciones de A codificadas con el codec de R | Rechazadas | `FrozenEpisodeCodec.encode`: la vista pertenece a otro contrato |
| Reanudar A con las piezas de R | Rechazada sin cambiar archivos | Ejecutor nativo |
| Reconstruir R desde el inicio de la fase en otra carpeta | Aceptada | Mismas observaciones que A y otro contrato. La sesión de A no cambia |

Una cohorte usa así una sola versión: la de su especificación de entrada, que fija edición, índice y representación, junto con el modelo, el codec, el prefijo y la regla de admisión que forman el contrato de la sesión. El ejecutor nativo guarda ese contrato en `identity.json` al crear la sesión y lo exige igual al reanudar.

## Regla cuando cambia una representación

Un cambio de representación, como otra revisión de los codificadores, deja las dimensiones iguales pero cambia la especificación, la huella del codec y el contrato. No hay una migración de estado: los episodios, los pesos rápidos y los estados por flujo de la representación anterior no se pueden reanudar ni copiar en la nueva. La sesión se reconstruye desde el inicio de su fase con la historia permitida, en otra carpeta, y la anterior se conserva intacta. La única herencia validada entre ediciones está en los datos: un vector se hereda solo con la misma identidad de codificador en FP32 estricto, y los textos de la v3 pasan un contraste bit a bit por activo ([edición v3.1](../../docs/data/edition-v3-1.md)).

Tampoco se hereda un calibrador. En la comparación por ventanas, la calibración conformal de cuantiles se ajusta para cada brazo, semilla y ventana con sus propias predicciones de calibración, y queda congelada con su huella antes de evaluar.

## Mutación dirigida

Se retiró por turno cada comprobación de producción implicada y se ejecutó la prueba completa con el enlace nativo.

| Comprobación retirada | Resultado |
| --- | --- |
| Codec con otra especificación en `TitansBinding.check` | Detectada |
| Fuente del prefijo en `FinancialSession` | Detectada |
| Identidad de entradas del lote en el predictor | Detectada |
| Contrato de entrada en el codec episódico | Detectada |
| Identidad del modelo en el contrato y en el ejecutor nativo | Detectada |
| Representación común de los activos de la cohorte | Detectada |
| Edición materializada del verificador de prefijos | Detectada |
| Fuente del corpus en `run_observation_source` | No detectada, mutante equivalente |

El último mutante es equivalente con artefactos reales: la identidad del índice incluye la de su corpus, así que la comprobación de la vista rechaza el mismo índice ajeno. Retirar solo la identidad del modelo del ejecutor nativo, sin tocar el contrato, tampoco basta para aceptar otra receta, porque el estado inicial que guarda el ejecutor lleva el contrato completo.

## Límites

- La prueba usa el ejecutor y el codec en CPU. Las comprobaciones de identidad no dependen del dispositivo, pero esta evidencia no ejecuta la ruta CUDA.
- No se ha añadido un módulo `memory/snapshot.py`. El contrato de versión ya está repartido entre `FinancialInputSpec`, la identidad del codec, el contrato de `FinancialSession` y la identidad del ejecutor nativo, y la prueba los comprueba juntos. Un módulo paralelo duplicaría esas identidades.
- La persistencia recuperable corresponde a MT-065 y la concurrencia a MT-063. Esta prueba solo comprueba compatibilidad y que un rechazo no escribe.
- Una identidad igual no demuestra equivalencia semántica. Solo garantiza que no se combinan piezas declaradas distintas.

```bash
MARS_TITAN_EPISODIC_NATIVE=<enlace compilado> uv run pytest tests/memory/test_bundle_compatibility.py
```
