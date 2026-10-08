# Candidatos de aprendizaje por refuerzo

KLPO terminal Full-KL es el candidato secuencial prioritario. PPO conserva tres
controles identificados y Double DQN aporta la referencia de valor con replay.
La elección delimita qué implementar y contrastar. No selecciona un ganador
empírico ni habilita aprendizaje mientras la edición histórica siga incompleta.

| Candidato | Motivo para conservarlo | Condición y límite |
| --- | --- | --- |
| PPO-Clip con diagnóstico KL, penalización adaptativa y Clip con parada KL | Permiten aislar cómo se controla el desplazamiento bajo el mismo actor, crítico y seis acciones. | La KL se mide después de la época. Recorte, penalización y parada no garantizan una mejora de cartera. Sus controladores tienen identidad propia. |
| Double DQN | Conserva una alternativa de valor que separa la selección de la acción y la valoración por la red objetivo. Su replay permite un primer control de reutilización. | La comparación registra exposición a transiciones y actualizaciones, además del tiempo. Replay no añade mercados independientes. |
| KLPO terminal Full-KL | Centra el score con una suma completa de seis acciones y utiliza el retorno de la trayectoria recogida, sin crítico para ese gradiente. | Exige trayectorias completas, una referencia histórica verificable y soporte común. La pérdida pura no integra todavía su recogida ni recuperación. |
| SAC discreto, condicionado | Calcula sumas sobre acciones y reutiliza transiciones con replay. | Solo se incorporará si se mide una limitación de reutilización que justifique actor, dos críticos, objetivos y temperatura adicionales. No está implementado en esta entrega. |

[PPO v2, secciones 3–5](https://arxiv.org/pdf/1707.06347v2) distingue las
pérdidas recortada y penalizada. La
[implementación documentada de Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
añade parada por KL aproximada. El control local usa la KL categórica completa
y una frontera posterior a la época, por lo que no se presenta como reproducción
literal de ese ejecutor. [Double DQN v3](https://arxiv.org/abs/1509.06461v3)
motiva separar selección y valoración, sin trasladar sus resultados de Atari a
mercados. [SAC discreto v2, sección 3](https://arxiv.org/pdf/1910.07207v2)
reemplaza estimaciones sobre acciones por sumas en su actor y críticos. Su
objetivo incluye entropía. Esa bonificación no es beneficio contable.

## KLPO y separación de tareas

El candidato secuencial usa [arXiv 2610.08963v1](https://arxiv.org/abs/2610.08963v1),
hipótesis 2.1–2.3 y ecuaciones 4.5–4.17, junto al
[pin 304e5ac7](https://github.com/yifanzhang-pro/KLPO/tree/304e5ac7ca573d45a0e42e6eaea95203260402bc).
Se conservan separados del [pin predictivo 30c0ae8c](klpo-review.md). No se
migran beta, auxiliares ni resultados anteriores. El PDF arXiv y el PDF posterior
del repositorio son versiones distintas.

Full-KL enumera la corrección bajo las seis probabilidades históricas. No conoce
el retorno de las acciones alternativas ni calcula una pérdida financiera
exacta. Su gradiente terminal coincide en población con el objetivo por
prefijos bajo la misma q que generó las acciones y continuaciones. La referencia
se mantiene fija durante el episodio y no se filtran trayectorias por resultado.
La igualdad del primer gradiente no implica igualdad de Hessianas de los grafos
con stop-gradient.

El [oráculo independiente](../../reports/engineering/terminal-klpo-oracle-20261008.json)
enumera 100 trayectorias finitas con seis acciones, longitudes distintas y ruido
intermedio y terminal. Confirma el gradiente poblacional y la sustitución del
retorno descontado por `gamma^u * G_u`. Usar recompensa inmediata, omitir ese
factor, dividir cada trayectoria por longitud o filtrar retornos cambia el
gradiente. Las [regresiones](../../native/tests/klpo_terminal_oracle.py) no llaman
a un entorno ni actualizan parámetros.

El ajuste predictivo de una decisión conserva padre sin adaptación, MAE,
pérdida esperada, REINFORCE y KLPO exacto/Full/MC. Al madurar la etiqueta se
conocen los errores de las 21 acciones. La suma completa elimina el muestreo de
esa expectativa, pero no garantiza generalización. El
[contraejemplo con el decoder real](klpo-quadratic.md) muestra que bajar F puede
empeorar el error de la mediana emitida.

Las ponderaciones también forman parte del contraste. El ejecutor
`training/predictive_run.py` utiliza `batch["weight"]`, mientras que
`posttraining/run.py::_loss` aplica una media simple. Un estudio nuevo debe
fijar pesos por sesión y compartirlos entre controles. Cambiar de ejecutor y de
ponderación a la vez no aísla el algoritmo. PPO, KLPO y Double DQN tienen objetivos
de optimización distintos. Comparten los datos admitidos, cortes, ejecución,
costes y evaluación externa fijados antes de comparar, con presupuestos y
denominadores registrados.

El [núcleo terminal](../engineering/terminal-klpo.md) es una función numérica
separada. Faltan el registro de trayectorias, su consumidor del actor y la
recuperación conjunta. El cierre de cinta, una ruina, la falta de valoración y
una pausa por recursos no son finales intercambiables. No se inventan precios,
recompensas ni liquidaciones para completar una trayectoria.
