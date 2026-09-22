# Plan de cómputo y campaña

Todavía no hay un calendario definitivo para la campaña. El equipo estará disponible para sesiones largas y recuperación mediante checkpoints. Esta disponibilidad no se convierte en una estimación de días de entrenamiento sin medir el flujo real.

## Decisión inicial de campaña

El piloto usa hasta 64 activos y un periodo de desarrollo que permita comprobar disponibilidad, referencias y consumo. La comparación principal se plantea con hasta 128 activos estadounidenses, seleccionados sin consultar sus resultados futuros. Se conservan varios periodos de evaluación y las repeticiones acordadas. La ampliación a 256, al universo completo o a otro mercado requiere una decisión posterior.

La prioridad es completar una comparación interpretable de memoria con referencias. Recurrencia entra primero con pocos pasos fijos. Una puerta compleja, replay paramétrico, destilación y otros codificadores necesitan un resultado de desarrollo que justifique su coste. No se entrenará el producto cartesiano de todos los métodos de la biblioteca.

## Medir antes de presupuestar

El piloto debe informar tiempo de preparación por GiB, tiempo de extracción por documento o imagen, muestras objetivo útiles por segundo, consumo máximo y tiempo de validación. Se distingue un registro leído, un token procesado y una predicción supervisada. Las ventanas solapadas pueden repetir cálculo sobre los mismos datos y no cuentan como observaciones nuevas.

Para cada ejecución se estimará:

$$T_{run}=T_{inicio}+N_{pasos}\,t_{paso}+T_{validacion}+T_{calibracion}+T_{guardados}.$$

El coste de campaña suma configuraciones, ventanas de evaluación y semillas, más preparación, cómputo de los modelos maestros de destilación, consolidación y repeticiones justificadas. Se utilizarán al menos una estimación central y otra conservadora obtenida de ejecuciones sostenidas. El coste se revisa si el tamaño cambia la saturación de GPU o convierte la lectura en cuello de botella.

Como ejemplo de cálculo, no como pronóstico del proyecto, 12 configuraciones con tres folds, tres semillas y 20 minutos por ejecución requieren 36 horas, antes de los costes auxiliares. Si cada ejecución tarda dos horas, la misma campaña exige 216 horas, equivalentes a nueve días ideales continuos. La disponibilidad 24/7 no evita esa multiplicación.

## Asignación inicial del presupuesto disponible

Cuando se mida el piloto, se reservará aproximadamente una cuarta parte del cómputo total para verificaciones, repeticiones justificadas y fallos. El resto se repartirá entre referencias, memoria y el contraste secundario que haya superado el piloto. Esa reserva no significa descartar resultados desfavorables para repetir hasta mejorar.

Se propone un cribado de hasta tres configuraciones y una semilla por familia candidata dentro del desarrollo. Se eligen antes las familias confirmatorias y se completan en ellas los folds y las tres semillas previstos. El límite anterior de diez configuraciones por familia se conserva como techo, no como una obligación de ejecutarlas. Se registran también decisiones manuales y variantes descartadas.

El test final permanece cerrado. Si falta tiempo, se eliminan extensiones y amplitud de búsqueda antes de reducir los controles de fuga, la comparación con referencias, la calibración o el análisis de errores. No se declara una mejora estable a partir de una sola ejecución elegida.

## Calendario relativo

| Momento | Trabajo prioritario | Condición para avanzar |
| --- | --- | --- |
| Inicio de implementación | Auditoría, piloto, referencias cero/lineal y recuperación. | Datos utilizables, ejecución retomable y estimación conservadora de tiempo. |
| Desarrollo principal | Codificador común y memoria mínima. | Entrenamiento verificable, estados correctos y comparación inicial completa. |
| Decisión de alcance | Revisar recurrencia, replay, familia moderna y extensión de cobertura. | Beneficio de desarrollo o pregunta útil que quepa en el presupuesto restante. |
| Campaña confirmatoria | Folds, semillas, calibración y evaluación financiera. | Protocolo cerrado, recursos reservados y test protegido. |
| Cierre | Reproducción de cifras, análisis crítico y redacción técnica. | Evidencias completas y comprobación de la reproducibilidad. |

La redacción acompaña todas las etapas. El calendario operativo sustituirá esta secuencia relativa cuando se conozcan la cobertura admisible y el coste medido de las ejecuciones.

## Sesiones largas en el portátil

Antes de una ejecución prolongada se comprobarán alimentación, suspensión prevista, ventilación, espacio libre y herramientas de observación. Se registrará el perfil de potencia que realmente tenga la RTX Max-Q. No se modifica automáticamente la configuración de energía ni se recomienda forzar límites térmicos.

La medida sostenida debe detectar reducción de frecuencia, crecimiento de colas, presión de RAM y escritura excesiva de checkpoints. Los trabajos se programarán de forma que no ejecuten dos entrenamientos pesados sobre la misma GPU por defecto. La preparación CPU y las transferencias podrán solaparse si el perfil muestra una mejora real.

Ante una interrupción se seguirá el [protocolo de recuperación](checkpoint-recovery.md). Los registros deben distinguir tiempo activo de GPU, espera de datos, pausas y tiempo de pared. Así se podrá ajustar la campaña sin confundir disponibilidad del ordenador con trabajo útil.
