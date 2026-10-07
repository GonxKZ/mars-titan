# Campañas temporales por mercado

El coordinador temporal admite un brazo `US` o `CN`. Antes de adquirir CUDA
contrasta el mercado de la configuración con el protocolo, los activos y los
mercados declarados en cada ventana. También verifica el alcance, los hashes,
las cuatro particiones y el presupuesto. Una edición parcial no puede
ejecutarse con una configuración que exija `full_corpus`.

La [configuración CN](../../configs/baselines/convergence-temporal-search-cn.json)
conserva los parámetros de la configuración estadounidense y cambia únicamente
el brazo. Se mantienen candidatos, semillas 42, 43 y 44, contexto de 64 sesiones,
criterio de validación, mínimos, paciencia y continuaciones emparejadas.
La ponderación es `natural` para todas las familias. La configuración programa,
en diez ventanas, 400 casos neuronales, 170 tabulares, 1.320 ajustes y
1.890 evaluaciones. La fiabilidad se calcula después sobre esos 1.890 modelos.
Estos recuentos son el diseño previsto, no ejecuciones CN completadas.

## Propagación y recuperación

Las etapas posteriores leen el mercado de la configuración conservada en la
referencia neuronal. No lo deducen del nombre del directorio. Todas las ventanas
deben declarar el mismo mercado, y cada ejecución debe pertenecer a ese brazo
con ponderación natural. Se rechaza, por ejemplo, una fila `US` dentro de una
referencia declarada `CN`, aunque existan otras filas chinas válidas.

El trabajador transmite el brazo a la cola tabular, los ajustes y la evaluación.
La interfaz independiente de `mars_titan.posttraining.heldout` acepta
`--arm US` y `--arm CN`, con `US` como valor predeterminado. El recibo de
evaluación fija el brazo en su identidad. La continuación también identifica
el mercado y el código que lo verifica. Los padres siguen emparejados por
semilla y los resultados confirmados se comprueban al reanudar.

Los cortes de cada mercado conservan su calendario y el siguiente cierre
admisible. Entrenamiento, validación, calibración y evaluación permanecen
separados. La reserva final no interviene en la selección. El brazo `US+CN`
sigue rechazado por este coordinador porque necesita un contrato para dos
calendarios y una representación contable común.

## Comprobación del recorrido

La prueba de integración genera desde cero un corpus técnico completo de un
activo, con 19 decisiones mensuales, las cuatro modalidades y 140 indicadores.
Las cifras y vectores son de prueba. No proceden de las empresas reales y no
se incorporan al corpus de investigación. La comprobación utiliza las guardas
reales de referencias, particiones y codificadores.

En CUDA se ejecutan 30 casos GRU de ese corpus CN, con una semilla, dos épocas
de búsqueda y controles de una época. Se solicita una pausa tras la primera
ventana. La reanudación termina las diez ventanas y conserva los hashes y las
fechas de modificación de los checkpoints confirmados. Después, el preflight
de las etapas siguientes reconoce CN a partir de los recibos reales generados
por esa ejecución técnica.

También pasan la comprobación existente de inferencia CPU/CUDA con tolerancias
`rtol=1e-5` y `atol=1e-7`, y la regresión de búsqueda estadounidense. Estos
ensayos comprueban ejecución y recuperación. No miden precisión predictiva
sobre empresas reales ni completan una campaña de las seis familias.

El [recibo de verificación](../../reports/resources/cn-temporal-campaign-20261007.json)
recoge pruebas, cobertura, mutaciones, tiempos y límites. La campaña
estadounidense conserva su runtime científico. La preparación china real
continúa ampliándose y mantiene su marca de cobertura parcial.
