# Materialización de cohortes por ventanas

La preparación y la codificación son etapas distintas. Un activo normalizado
puede no aportar ninguna muestra si sus modalidades no coinciden en el tiempo.
La codificación recorre todos los candidatos de una preparación completa y
conserva tanto los activos admitidos como las exclusiones y los errores.

Cada muestra contiene precios pasados, texto, hechos contables y un gráfico
regenerado únicamente con esos precios. El contexto macro es obligatorio. Las
ventanas principales tienen 64 sesiones consecutivas. Una sesión ausente rompe
la ventana, no se rellena con una observación de otro día. El texto se consulta
en las cinco últimas sesiones, con límites inclusivos de disponibilidad.

## Memoria y representaciones

`NewsWindows` conserva como máximo dos grupos de noticias. Cada grupo tiene un
límite de 16 MiB antes y después de leerlo. El texto conserva su diccionario
Arrow y se entrega registro a registro, sin expandir de golpe las cadenas
repetidas del grupo. No se acumula el historial textual del activo. El cursor
contable avanza por publicaciones y conserva las reglas de revisión y
ambigüedad de la referencia. Los vectores macro se calculan una vez por fecha
y se comparten entre los activos del mismo mercado.

Los precios y los hechos se leen por activo, con límites de 200.000 filas y
64 MiB por tabla. Los factores empresariales tienen otro límite explícito de
200.000 registros. Esos límites no son una selección de empresas. Superarlos
produce un error visible y exige revisar la partición, sin entrenar con un
recorte silencioso. El presupuesto del índice macro es una estimación que
incluye el vector y espacio para sus índices. No sustituye la medición del
pico de memoria del proceso.

La media textual utiliza todos los eventos admitidos en la ventana. Suma los
vectores en float64 y guarda la media en float32. Conserva el número de eventos,
su tipo y una huella del conjunto ordenado. La trazabilidad se reconstruye
desde el Parquet de noticias confirmado y los límites de la ventana.

El esquema estadounidense con ocho conceptos contables, siete ratios y
140 indicadores macro contiene 1.361 valores float32 por muestra, 5.444 bytes
numéricos antes de compresión. Esa cifra no incluye metadatos, estructuras Arrow,
memoria del codificador ni precios. Las ventanas de precios se construyen bajo
demanda para no guardar repetidamente sus solapamientos. Los conceptos y
unidades se conservan en los manifiestos. No se aplica una conversión de
conceptos o moneda entre mercados por coincidencia de dimensiones.

## Integridad y recuperación

Cada activo tiene una configuración inmutable, una tabla confirmada y un recibo
con huellas. La escritura del Parquet es atómica. Si se interrumpe, la caché de
vectores ya calculados permite repetir el activo sin presentar la tabla parcial
como válida. La recuperación comprueba identidad, cohortes, recuentos y
artefactos. Un error de entrada queda registrado por activo. Un fallo del
codificador interrumpe la ejecución para poder diagnosticarlo.

La versión 2 del manifiesto de supervisión identifica la cohorte en cada activo,
en las filas de muestras y etiquetas y en los lotes entregados al entrenador.
La cohorte `original_audited` conserva su etiqueta de auditoría interna. No se
convierte en `externally_verified` al codificar o entrenar. Las ediciones
históricas de versión 1 mantienen su contrato anterior.

Una cobertura completa significa haber recorrido todos los candidatos, no que
todos tengan datos utilizables. Solo se confirma la edición al terminar sin
errores. Todavía necesita etiquetas y factores del mercado correspondiente
antes de entrenar. La reserva desde 2024 permanece cerrada. Los codificadores
preentrenados actuales no acreditan una simulación histórica de sus propios
pesos, aunque las entradas respeten las fechas de disponibilidad.
