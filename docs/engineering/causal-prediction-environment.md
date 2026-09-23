# Entorno de refuerzo predictivo

`CausalPredictionEnv` recibe juntas las decisiones de todos los activos con
el mismo instante de predicción. Ordena sus identificadores antes de entregar
la observación. Así, el orden de las filas de un archivo no decide qué activo
obtiene primero una actualización. El entorno no entrena el predictor ni
actualiza sus pesos entre acciones de una misma cohorte.

La observación contiene precios, texto, gráficos, fundamentales, macro y una
máscara de activos presentes. La máscara distingue las filas reales del relleno
necesario para conservar un espacio Gymnasium de tamaño fijo. Las acciones de
las filas de relleno no producen decisiones ni recompensas.

La [interfaz de Gymnasium](https://gymnasium.farama.org/api/env/)
se implementa con `reset` y `step`, observación y recompensa, y señales separadas
de terminación y truncamiento. El entorno termina al consumir la fuente y
resolver las etiquetas pendientes. Una interrupción del entrenamiento debe
guardar el estado y reanudarlo, no cambiar el final del episodio.

## Reloj y etiquetas

El proveedor entrega una cohorte por llamada, con un índice entero recuperable.
Cada cohorte contiene identificadores, instante de predicción, un límite superior
de disponibilidad de sus entradas, etiquetas y fechas de maduración. El proveedor
debe justificar ese límite con los manifiestos de preparación. El entorno no
deduce una fecha de publicación a partir de un periodo contable.

Las entradas deben estar disponibles en la decisión. Las etiquetas deben madurar
después. Los datos de entrenamiento no pueden cruzar el inicio de 2023. La
validación usa decisiones y etiquetas de 2023. Las fechas de 2024 en adelante
se rechazan antes de decidir.

Una llamada a `step` registra todas las acciones de la cohorte. Después avanza
al siguiente instante y devuelve únicamente los créditos cuya etiqueta ya ha
madurado. Los créditos se ordenan por maduración e identificador del evento.
El último paso avanza hasta la última maduración pendiente. No inventa una
observación futura para seguir el episodio.

La información de un crédito identifica su acción, predicción, objetivo,
maduración y recompensa. No aparece en la observación anterior. Una política
con actualización en línea debe aplicar los créditos después de resolver
toda la cohorte, nunca entre empresas del mismo instante.

## Rejilla y objetivo

`ActionGrid` fija 21 retornos posibles con las etiquetas de entrenamiento.
La amplitud es el máximo valor absoluto de los percentiles 1 y 99, con
interpolación lineal. La rejilla es simétrica e incluye cero. La escala es
el promedio del valor absoluto de las etiquetas. Ambos valores tienen un
suelo de `1e-8` para el caso degenerado en que todas las etiquetas son cero.
La cola fuera de la rejilla se contabiliza por separado, sin recortar etiquetas.

Para una acción con retorno `a`, objetivo `y` y escala `s`, la recompensa es
`−|a − y| / s`. Como se conocen las pérdidas de las 21 acciones, también se
calcula el control de pérdida esperada exacta `Σ π(a|x) |a − y| / s`. El
refuerzo tendrá que compararse con ese control y con continuación supervisada,
no asumir una ventaja por utilizar recompensas.

La salida puntual para MAE es la mediana inferior de la distribución. No se
sustituye por su media. Se admite un error de suma de probabilidades de hasta
`1e-6`, seguido de normalización, para aceptar el redondeo de un softmax en
`float32`. Las distribuciones negativas, no finitas o con masa incoherente
se rechazan.

Las sumas acumuladas próximas a la mitad se recalculan con `math.fsum` antes
de resolver la mediana. El margen detecta un posible error de acumulación,
pero la comparación sigue siendo con 0,5. No desplaza ese umbral para tratar
como empates distribuciones cercanas que no lo son.

La recompensa escalar de Gymnasium es el promedio de los créditos maduros
de ese paso. Para entrenar el bandido predictivo, cada crédito debe asociarse
a su acción mediante `event_id`. No debe asignarse ese promedio a la última
acción si los créditos corresponden a cohortes anteriores.

## Presupuesto y recuperación

El espacio permite hasta 4.096 activos por cohorte. El bloque de observación
tiene por defecto un límite de 64 MiB y la cola admite hasta 65.536 etiquetas.
Una entrada que supera el presupuesto falla sin avanzar el estado confirmado.
Estos límites no autorizan omitir empresas. Si una edición admisible los supera,
debe revisarse la representación antes de ejecutar esa edición.

Se copia un bloque para que modificar un arreglo recibido no altere después
el estado del entorno. El cálculo de percentiles de la rejilla admite hasta
16.777.216 etiquetas, con un máximo de 128 MiB por representación `float64`.
Los temporales del cálculo y la memoria del llamador son adicionales. Ese
cálculo no necesita cargar las modalidades completas en RAM.

`snapshot` exporta tipos JSON simples con identidad, cursor, reloj, cohorte
actual, etiquetas pendientes y generadores aleatorios. `restore` comprueba
la identidad y la huella de la cohorte actual antes de cambiar el estado.
El contenedor de checkpoint del entrenamiento debe verificar la integridad
del archivo completo. El entorno no incorpora pesos, optimizador ni el corpus
dentro de su estado.

La comprobación de Gymnasium avisa de que el espacio numérico no tiene cotas
estadísticas fijadas. Las entradas reales sí deben ser finitas y representables
en `float32`. No se inventan cotas de características para ocultar ese aviso.

## Alcance implementado

Las pruebas actuales usan proveedores sintéticos y cubren maduración, orden,
recuperación, particiones y límites. El adaptador del corpus Parquet, los
entrenadores de refuerzo y las campañas científicas requieren su propia
integración y evidencia. Este entorno no ejecuta operaciones financieras
ni implementa PPO o MARS-TITAN.

La [verificación local](../../reports/resources/causal-environment-quality.json)
registra cobertura, complejidad, CRAP y mutaciones dirigidas. El
[ensayo de coste](../../reports/resources/causal-environment-benchmark.json)
usa 2.500 activos por cohorte, tres repeticiones y 64 pasos medidos después de
ocho de calentamiento. Solo mide el entorno CPU con entradas sintéticas en RAM,
no la lectura desde disco ni los pasos del modelo.

En el AMD Ryzen 9 8945HS, las medianas de las tres repeticiones fueron
21,0, 24,1 y 16,6 ms por cohorte. El máximo RSS del proceso fue 168.456.192
bytes, incluidos la fuente sintética y el calentamiento. La fuente ocupaba
16.810.000 bytes de arreglos. La materialización del corpus y otra aplicación
seguían activas. La variación observada no permite presentar esas cifras como
una latencia garantizada ni convertirlas en tiempo de entrenamiento.
