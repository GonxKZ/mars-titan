# Escritura episódica por error maduro

`MatureErrorBank` implementa la variante `episodic_m2_three_index_v1`. Compone tres bancos nativos v2 con cupos 50/25/25 y devuelve su unión deduplicada. Esta primera pieza no conecta todavía M2 a `FinancialSession`. M permanece desactivado y no se define la sorpresa completa M3.

`MatureErrorConfig.capacity` es el único presupuesto total de slots, denominado `B_mem`, entre 4 y 1.024. Los cupos parten de los cocientes enteros de `(2·B_mem, B_mem, B_mem)` entre cuatro. Las plazas restantes se asignan por mayor resto, con desempate reservorio, selectivo y reciente. Para 1.024 slots corresponden 512, 256 y 256.

Todos los candidatos maduros se ofrecen a los tres índices. El reservorio reutiliza el algoritmo nativo v2 y su semilla explícita. El selectivo conserva los mayores valores `abs(error)`, con desempate por menor ID. No reemplaza un residente por un candidato de menor puntuación. El reciente conserva los últimos IDs del orden maduro. El error recibido es firmado y FP64. Su procedencia debe comprobarla el coordinador contra la predicción realmente emitida. La puntuación no se estima con una regresión ni se normaliza con el futuro.

Cada propuesta modifica copias y devuelve un banco nuevo. Los originales permanecen intactos ante errores de datos, archivo o presupuesto. El snapshot conserva la identidad, los tres archivos nativos, las puntuaciones selectivas y un recibo. La restauración exige el mismo ámbito y cursor en los tres bancos, contrasta cupos, tipos, IDs, puntuaciones, tamaños y copias de cada episodio. Un mismo ID tiene que conservar exactamente sus valores y bits en todos los índices.

La capacidad física y el número de episodios únicos son distintos. Los tres bancos pueden almacenar copias de un episodio, siempre dentro de `B_mem` slots totales. La consulta recibe cada ID una vez. No se rellenan los solapamientos con candidatos adicionales. El recibo informa slots, unión, duplicados, ofertas, puntuaciones, nuevas entradas, expulsiones y bytes de archivo. Ofrecer un candidato a tres índices se cuenta como tres ofertas.

El componente limita cada propuesta a 8.192 candidatos y estima conjuntamente registros, copias, archivos y buffers con un máximo de 64 MiB. La estimación no certifica el asignador de Python o LibTorch. El [recibo CPU](../../reports/engineering/mature-error-bank-20261009.json) distingue el pico trazado de Python, el RSS del proceso y el tamaño de los archivos. No se crea una matriz de distancias.

Las pruebas usan etiquetas y errores manuales. Cubren todos los cupos admitidos, signos y empates, errores pequeños que necesitan FP64, reservorio frente al nativo, recuperación exacta, copias divergentes y presupuestos. No ejecutan modelos, ajuste de etiquetas ni evaluación financiera.

La integración pendiente debe conservar la emisión de toda la cohorte antes de resolver etiquetas, comprobar el error original y publicar los tres índices con el resto del estado. Reutilizará el Executor y sus artefactos. Las posibles consolidaciones M sobre empates o fronteras de puntuación requieren contratos separados. No se han activado.
