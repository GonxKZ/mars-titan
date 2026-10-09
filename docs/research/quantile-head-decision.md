# Propuesta sobre la cabeza de cuantiles

Estado: propuesta registrada el 9 de octubre de 2026 para [#22](https://github.com/GonxKZ/mars-titan/issues/22). No está implementada en ningún modelo ni aprobada. Debe decidirse antes de entrenar, cuando se levante el bloqueo de aprendizaje, porque cambia la salida y la identidad de varias arquitecturas.

## Estado comprobado en el código

| Modelo | Salida | Pérdida definida hoy | Dónde se comprueba |
| --- | --- | --- | --- |
| GRU candidata nativa | Cinco cuantiles (0,025, 0,1, 0,5, 0,9 y 0,975). Mediana libre e incrementos con softplus, orden no decreciente por construcción | Ninguna. No hay pinball en `src/` ni en `native/` | `Candidate::quantiles` en `native/src/candidate.cpp`. `CandidateInputAdapter.forward` devuelve `quantiles[:, 2]` como mediana |
| Titans-MAC (`transformer_direct`, `mac_disabled`, `mac_frozen`, `mac_online`) y MARS-TITAN | Escalar, `nn.Linear(D, 1)` heredada de `MultimodalReference` (`output="scalar_corpus_target"`) | Ninguna. `FinancialPredictor` no tiene todavía bucle de entrenamiento en `develop` | `models/titans/financial.py` |
| Lectura episódica de MARS-TITAN | Aplica la cabeza recibida al estado refinado y exige una salida escalar por flujo | Depende del entrenador futuro | `apply_episodic_readout` en `models/titans/episodic_readout.py` |
| Transformer compacto, RNN, LSTM, GRU y DLinear de referencia | Escalar, `nn.Linear(D, 1)` | MSE, MAE o Huber según la configuración. `reference_design.py` alterna las tres | `models/baselines/multimodal.py`, `training/reference_run.py` |
| Ridge | Escalar | Error cuadrático penalizado | `models/baselines/ridge.py` |
| XGBoost externo | Escalar | `reg:squarederror` | `models/baselines/external_boosting.py` |
| Controles cero y padre | Escalar | No se ajustan | `evaluation/prediction_statistics.py` |

La cabeza de cuantiles de la GRU existe, pero ninguna ruta del repositorio la ajusta todavía. Las demás arquitecturas solo pueden ofrecer intervalos mediante un radio simétrico calibrado (`forecast_reliability.py`), igual para todas sus filas.

## Por qué afecta a la comparación

La métrica primaria es el MAE residual por sesión. Bajo error absoluto, la predicción puntual óptima es la mediana condicional ([Gneiting, 2011](https://doi.org/10.1198/jasa.2011.r10138)). Una cabeza ajustada con pinball y evaluada en su mediana optimiza lo mismo que se mide. Un modelo ajustado con MSE estima la media condicional. Con residuos asimétricos o de colas gruesas, media y mediana difieren y ese modelo parte con desventaja en MAE sin que su arquitectura tenga que ver. Si solo la GRU se ajustase con pinball, una diferencia de MAE mezclaría arquitectura y objetivo de ajuste.

El efecto también puede ir en contra de la cabeza de cuantiles. Con la media de pinball sobre cinco niveles, la mediana recibe $0{,}5\,|u|/5=|u|/10$ por fila, una décima parte del peso que tendría con una pérdida L1 escalar, y comparte representación con las colas. Puede ayudar como regularización o perjudicar al MAE. Ese efecto debe medirse.

La hipótesis H4 del [protocolo](protocol.md) pregunta si la incertidumbre ayuda a abstenerse. La curva riesgo-cobertura de `forecast_scores.selective_risk` necesita una incertidumbre que cambie entre filas. Un radio constante da cobertura y anchura, pero no ordena la abstención. Comparar H4 entre arquitecturas exige que todas emitan el mismo tipo de incertidumbre.

## Opciones

**A. Salida escalar común con pérdida L1.** Todas las arquitecturas predicen un escalar ajustado con error absoluto. La GRU usaría solo su mediana con pérdida L1, lo que equivale a una cabeza escalar porque los incrementos no reciben gradiente. Es el cambio mínimo y alinea objetivo y métrica. H4 queda sin contraste entre arquitecturas y los intervalos se limitan al radio constante.

**B. Misma cabeza de cuantiles para todas las arquitecturas neuronales.** GRU, Transformer compacto, Titans-MAC, MARS-TITAN y CM-v1, que hereda el núcleo, usan la misma especificación de niveles, parametrización y pérdida. La mediana es la predicción puntual del MAE. Las referencias tabulares conservan su salida escalar.

**C. Cuantiles solo donde ya existen.** La GRU conserva su cabeza y las demás arquitecturas siguen escalares. La calibración, la cobertura y la abstención se informan como resultado secundario de la GRU. Es la regla vigente de la reconciliación del 8 de octubre si no se añaden cabezas.

| Criterio | A | B | C |
| --- | --- | --- | --- |
| Objetivo de ajuste coherente con el MAE | Sí, en todas | Sí, mediante la mediana | Depende de la pérdida de cada familia |
| Arquitectura separada del objetivo | Sí | Sí, con el control de cabeza | No. La GRU cambia ambos a la vez |
| H4 comparable entre arquitecturas | No | Sí | No. Solo la GRU |
| Cambios en modelos e identidades | Pérdida y GRU | Cabeza común, pérdida, lectura episódica e identidades | Ninguno |
| Coste de cálculo añadido | Ninguno | Cabeza de 5D parámetros y cinco términos de pérdida por fila | Ninguno |
| Riesgo principal | Se pierde la pregunta de abstención | La cabeza puede empeorar el MAE | Comparación principal confundida por el objetivo |

## Propuesta

Se propone la opción B con estas condiciones.

1. **Especificación única `quantile_head_v1`.** Niveles 0,025, 0,1, 0,5, 0,9 y 0,975. Mediana libre e incrementos con softplus, como la GRU nativa. Pérdida igual a la media de pinball con pesos iguales. Predicción puntual igual a la mediana. La implementación PyTorch debe ser un único módulo reutilizado por `MultimodalReference` y `FinancialPredictor`, con una prueba de paridad numérica frente a `Candidate::quantiles` en FP32 y FP64 con los mismos pesos.
2. **Identidades nuevas.** Cada variante con la cabeza recibe identidad propia. Las variantes escalares se conservan con su código y evidencia como referencia histórica. La paridad de MARS-TITAN con C y M apagados se comprueba con la misma cabeza.
3. **Control de la cabeza.** En una arquitectura barata, el Transformer compacto, se ajustan la cabeza escalar L1 y `quantile_head_v1` con las mismas filas, semillas y presupuesto. El contraste es `delta(escalar_l1, cuantiles)` del MAE por sesión con intervalo simultáneo por bloques. Si la cabeza de cuantiles empeora el MAE de forma que el intervalo excluya el cero, se descarta B para la comparación principal y se pasa a A, conservando los cuantiles como experimento secundario. Si no hay diferencia concluyente, la cabeza se mantiene y se informa como factor fijado.
4. **Referencias tabulares.** Ridge y XGBoost siguen escalares en la comparación principal, con su pérdida registrada. Como sensibilidad, la versión fijada de XGBoost (3.3.0) admite `reg:absoluteerror` y `reg:quantileerror` con varios `quantile_alpha` según su [documentación de parámetros](https://xgboost.readthedocs.io/en/stable/parameter.html). Ridge no tiene un análogo directo y barato.
5. **Calibración común.** Todos los modelos con cuantiles usan el mismo procedimiento, ajustado con su tramo de calibración y congelado antes de evaluar: una corrección por intervalo central al estilo de la regresión cuantílica conformalizada ([Romano, Patterson y Candès, 2019](https://arxiv.org/abs/1905.03222)). Los modelos escalares reciben el radio simétrico existente para informar cobertura y anchura, sin curva riesgo-cobertura. Ninguna cobertura se presenta como garantizada bajo dependencia temporal.
6. **Métricas.** Pinball por nivel, frecuencia por nivel, cobertura, anchura, afirmaciones de signo y riesgo-cobertura, ya implementadas en `evaluation/forecast_scores.py`, se informan antes y después de calibrar. La decisión principal sigue siendo el MAE por sesión de la mediana, con las comparaciones de `evaluation/paired_comparisons.py`.

## Consecuencias si se aprueba

- Cambian la salida y la identidad de Titans-MAC, MARS-TITAN, CM-v1, el Transformer y las referencias neuronales. Todos se reentrenarán desde cero con la edición histórica, así que no se pierde ningún resultado comparable.
- `apply_episodic_readout` comprueba hoy que la cabeza devuelva un escalar por flujo. Esa comprobación debe aceptar la forma `[flujos, 5]` y entregar la mediana como predicción puntual.
- La búsqueda de tasas de aprendizaje debe repetirse porque cambia la escala del gradiente de la mediana.
- Los controles cero y padre y las referencias tabulares solo entran en métricas puntuales y en el radio simétrico.
- El coste de la cabeza no está medido. Se espera pequeño frente a los codificadores, pero debe medirse en la integración.
- Si se rechaza B, la opción C mantiene válida la comparación principal siempre que cada familia declare su pérdida, y limita H4 a la GRU como resultado secundario.

## Alcance de este documento

No implementa cabezas, pérdidas ni calibradores en los modelos y no ejecuta entrenamiento. Las rutas que cambiarían son existentes (`models/baselines/multimodal.py`, `models/titans/financial.py`, `models/titans/episodic_readout.py`, `training/reference_run.py`) o futuras (un módulo común de cabeza y pérdida y `src/mars_titan/calibration/`). No se afirma que ninguna opción mejore la predicción.
