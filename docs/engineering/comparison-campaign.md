# Campaña comparativa sobre el corpus completo admisible

Estado: especificación técnica de la campaña. No acredita entrenamientos completos
ni sustituye las evidencias de las sondas anteriores. Fecha: 22 de septiembre de 2026.

## Alcance

La campaña recorrerá la copia completa de FinMultiTime y entrenará las referencias
sobre todas las muestras admisibles del universo declarado, sin el límite anterior
de 64 o 128 activos. Se mantendrán separados Estados Unidos y China. MARS-TITAN y
sus ablaciones M0 a M3 quedan fuera de esta campaña previa.

Completo describe la cobertura de las particiones asignadas a cada modelo, no el
uso indiscriminado de cualquier archivo. Se conservarán originales, duplicados
identificados, exclusiones, registros pendientes y procedencia. No se declarará
completo un entrenamiento sobre las 65 muestras conocidas antes de terminar la
auditoría del resto del universo.

Los codificadores de texto e imagen permanecerán congelados al principio. Las
referencias aprenderán con las cuatro modalidades y el contexto macro. El uso de
pesos publicados después del periodo estudiado conserva la limitación descrita en
[procedencia de los codificadores](../data/pretraining-audit.md), sin presentarlo
como simulación histórica estricta de la disponibilidad del modelo.

## Tres niveles de cobertura

1. **Archivo presente.** Existe una fuente atribuible al instrumento. No demuestra
   fechas válidas, contenido correcto ni coincidencia temporal con otras fuentes.
2. **Registro admisible.** Su identidad, contenido, unidad, versión y disponibilidad
   cumplen el contrato correspondiente. Un registro desconocido no se da por válido.
3. **Muestra supervisada utilizable.** Coinciden una ventana de precios válida,
   noticias completas verificadas, hechos contables publicados, un gráfico generado
   con pasado, contexto macro conocido y una etiqueta madura del tramo permitido.

El inventario registra 216.453 archivos y 116.185.016.265 bytes. La revisión de
rutas identifica 2.639 instrumentos estadounidenses y 810 chinos con las cuatro
fuentes originales. Estas cifras son presencia de archivos, no cohortes listas
para entrenar. Los [resultados de preparación existentes](../../reports/data/news-coverage-expansion.md)
solo acreditan 65 muestras estrictas de dos activos.

La revisión detectó que el inventario interpreta algunas carpetas de imágenes
como símbolos por asumir una profundidad fija. La corrección debe contrastar
carpeta del activo y nombre del PNG, mantener los hashes de los originales y
generar una revisión nueva del inventario. La huella de un archivo no valida su
clasificación semántica.

El selector técnico también filtra por sector actual. Esa condición no gobernará
la campaña completa. La ausencia de sector no elimina una empresa si las entradas
requeridas son admisibles. El sector actual se conserva como metadato descriptivo,
no como clasificación histórica acreditada.

## Regla de admisión y reserva temporal

Para el activo $i$ y la decisión $t$, la admisión exige

$$
e_{i,t}=\mathbf{1}[P_{i,t}\land N_{i,t}\land F_{i,t}\land G_{i,t}]
\mathbf{1}[M_t]\mathbf{1}[Y_{i,t}\text{ pertenece al tramo permitido}].
$$

$P$ representa 64 sesiones consecutivas de precios válidos. $N$ exige texto
editorial completo verificado, con relación comprobable con el activo, dentro de
las cinco sesiones de la ventana de noticias. $F$ conserva el mínimo inicial de
al menos un hecho observado, finito y no ambiguo entre los conceptos de balance
de la representación existente. $G$ es el gráfico regenerado exclusivamente desde
esa ventana de precios. $M$ exige al menos un indicador macro observado antes de
la decisión, no únicamente una rejilla de posiciones ausentes.

La presencia de cuatro modalidades no significa que las 140 variables macro o
todos los conceptos contables estén observados. Cada variable conserva máscara,
antigüedad y motivo de ausencia. No se inventan valores para completar canales.
Los ocho conceptos de balance iniciales están definidos en
`src/mars_titan/data/samples.py`. Ampliar sus variables o endurecer el mínimo de
admisión exige otra versión del catálogo y una comparación sobre muestras comunes,
sin cambiarlo después de observar el test.

La revisión del texto tiene que distinguir fecha declarada, publicación
verificable, modificaciones posteriores y contenido atribuido a otra empresa.
Una coincidencia de símbolo o un título correcto no verifica el cuerpo completo.
No se sustituye esta comprobación por un clasificador de relevancia sin evaluar
sus errores. Los registros no verificables se mantienen fuera del contraste estricto.

La campaña de desarrollo conserva decisiones hasta 2023 y mantiene cerrada la
reserva de 2024 en adelante. La inspección de archivos posteriores para inventario
no permite seleccionar variables, empresas o modelos por su cobertura futura.
Las comparaciones previas usan validación cronológica. La evaluación final común
se ejecutará cuando también esté fijado el candidato MARS-TITAN.

La comparación posterior exige también los mismos datos de entrenamiento para
el candidato. Si MARS-TITAN no pudiera ajustarse sobre ese universo, se definirá
otra comparación y se volverán a ajustar las referencias sobre la misma población.
Recortar solo las filas de evaluación de una referencia entrenada con más datos
no elimina esa diferencia de información.

No se mezcla todo el periodo para ajustar normalización, PCA, residualizadores,
hiperparámetros o umbrales. Cada fold utiliza el pasado autorizado, purga etiquetas
que crucen fronteras y conserva identificadores estables de sus muestras.

## Contexto macro y variables de empresa

La [cesta macro](../data/macro-catalog.md) contiene 140 indicadores: 70 originales
y 70 fórmulas. Los artefactos actuales tienen valores para 125 indicadores en alguna
fecha. Las 420 posiciones del tensor son valores transformados, máscaras y edades.
Los dos mercados aplican sus calendarios a una cesta global, no a dos colecciones
independientes de 140 indicadores nacionales.

Se calculará cada indicador una vez por mercado, decisión y versión de las fuentes.
Las muestras de empresas lo referenciarán mediante una clave común. Una observación
mensual o trimestral no se convierte en información intradiaria por repetir su
último valor conocido. Su frecuencia y antigüedad permanecen visibles.

Los fallos de adquisición, identificadores pendientes y ausencia de vintages se
resolverán en la fuente cuando sea posible. Si no hay evidencia suficiente, el
resultado seguirá ausente. Se revisará además el intervalo necesario para las
ventanas de entrenamiento, incluida la historia anterior que requieren los retardos.

Los factores empresariales son una capa distinta. El catálogo debe inventariar
los conceptos contables realmente disponibles, sus unidades, tipo de magnitud,
periodos y revisiones, sin restringir la auditoría a las ocho posiciones de la
sonda. El conjunto de variables de cada comparación se congela con desarrollo.

Como primera familia verificable se proponen estos cocientes. A es activo total,
CA activo corriente, L pasivo total, CL pasivo corriente, E patrimonio, AR cuentas
por cobrar y AP cuentas por pagar.

| Variable | Fórmula | Condición adicional |
| --- | --- | --- |
| Liquidez corriente | CA / CL | CL positivo |
| Fondo de maniobra relativo | (CA − CL) / A | A positivo |
| Pasivo sobre activo | L / A | A positivo. No se denomina deuda financiera |
| Patrimonio sobre activo | E / A | A positivo. Se conserva patrimonio negativo |
| Cuentas por cobrar sobre activo | AR / A | A positivo |
| Cuentas por pagar sobre activo | AP / A | A positivo |
| Pasivo sobre patrimonio positivo | L / E | E positivo. El resto conserva una causa de ausencia |

Los componentes deben tener unidad compatible, el mismo cierre contable y una
publicación y versión coherentes. Inicialmente se exige la misma presentación.
La disponibilidad del derivado es el máximo de las disponibilidades de sus
componentes. El cálculo precede a la transformación logarítmica. No se divide
directamente el vector actual, que puede mezclar cierres distintos por concepto.

Los flujos de caja requieren intervalos iguales y distinguir trimestre, acumulado
y año completo. Una variación de caja no sustituye su saldo. No se calculan ROA,
ROE, márgenes, PER o deuda financiera neta sin sus componentes verificables. Las
razones de exclusión permiten ampliar el catálogo después sin ocultar estas carencias.

## Almacenamiento y lectura

Se reutilizan PyArrow, los escritores atómicos, las huellas, la caché y las reglas
de disponibilidad. Polars se evaluará para proyección, agregación y uniones por
fecha sobre tablas columnares. La equivalencia se comprueba sobre claves, valores,
ausencias y orden. Usar una API lazy no demuestra que todas sus operaciones tengan
memoria limitada. Se revisará el plan y el pico del proceso.

Los originales no se mueven. Se normalizan por instrumento o partición acotada y
se escriben Parquet con Zstandard. Las ventanas solapadas se construyen al leer.
Texto, gráficos y hechos comparten representaciones por identidad y versión cuando
corresponde, sin confundir contenido igual con disponibilidad igual.

El esquema separa índices de muestras, representaciones modales, contextos macro
y etiquetas. La clave macro evita duplicar 420 valores por empresa y sesión.
Los lectores entregan lotes con límites de filas y bytes antes de convertir a
objetos Python o tensores. No hay una llamada final que concatene todo el corpus.

Para $B$ ejemplos, $F$ valores y $b$ bytes por valor, la entrada numérica ocupa
$BFb$ bytes. Con los 1.660 valores de las sondas, un lote de 64 ocupa 424.960 bytes
en float32, sin etiquetas. Esa cuenta no incluye activaciones, optimizador,
buffers, índices ni el contexto CUDA. El presupuesto debe medirlos por separado.

## Etiquetas y fundamentos numéricos

Se conserva la etiqueta residual del [protocolo](../research/protocol.md):

$$
y_{i,t}=r^{OC}_{i,t+1}-\hat\alpha_{i,t}-\hat\beta_{i,t}r^{OC}_{m,t+1}.
$$

El retorno futuro de mercado es parte de la etiqueta, nunca una entrada. Los
coeficientes usan hasta 252 sesiones y al menos 126 pares ya observados. Una
implementación por estadísticas móviles puede evitar recalcular cada ventana,
pero deberá conservar las exclusiones y coincidir con la referencia float64.
Se prueban huecos, publicaciones tardías, varianza nula y cancelación numérica.

Ridge acumula estadísticas suficientes por bloques. Su memoria principal es
$O(F^2)$, independiente del número total de filas, pero su coste de cálculo no lo
es. Se resuelve el sistema regularizado sin invertir una matriz explícitamente.
Reutilizar estadísticas para varios valores de regularización exige el mismo fold
y la misma transformación. SGD, si se añade, conserva otro identificador de modelo.

China necesita su propio índice de mercado y un contrato de etiqueta comprobado.
No se reutiliza SPY ni se trasladan automáticamente las convenciones económicas
estadounidenses. La falta de fundamentales con publicación acreditada sigue
impidiendo una muestra china estricta, aunque existan precios y contexto macro.

## Referencias y formas de ejecución

| Referencia | Recorrido de datos | Comprobación necesaria |
| --- | --- | --- |
| Cero y media histórica | Acumuladores con corte temporal | La media no incorpora etiquetas inmaduras |
| Ridge | Estadísticas por bloques y resolución regularizada | Paridad numérica y condicionamiento |
| Boosting | HistGradientBoosting si cabe. XGBoost externo con otro identificador si se activa | Matriz externa real, cachés medidas y diferencias metodológicas declaradas |
| GRU | Ventanas independientes, estado reiniciado | Todas las modalidades influyen en el ajuste |
| DLinear adaptado | Descomposición dentro de la ventana disponible y fusión multimodal | Mismo objetivo residual, sin suavizar etiquetas con futuro |
| TCN compacto | Convoluciones temporales sobre la ventana y fusión multimodal | Campo receptivo y presupuesto documentados |
| Memoria asociativa de referencia | Estado de pesos rápidos con regla delta | Cronología, estabilidad y recuperación de estado |

La referencia asociativa utiliza la lectura $A^Tq$ y la familia de actualización
$A'=(1-\lambda)A+\eta k(v-A^Tk)^T$, con claves y valores de dimensión y versión
fijadas. Si el valor incorpora un retorno, debe estar maduro. Las escrituras se
aplican después de la cohorte en orden canónico. No se presenta como reproducción
completa de Titans ni como el candidato MARS-TITAN.

No se afirmará que HistGradientBoosting entrenó con todo si se usó una muestra.
Un backend que no cabe conserva el estado de ejecución incompleta y su causa.
Adoptar otro algoritmo requiere comparar y nombrar esa alternativa, no cambiarla
silenciosamente. La memoria externa tampoco elimina etiquetas, histogramas,
cachés y estructuras auxiliares.

Las formas de ejecución comparables son lectura por bloques, configuración de
trabajadores y, para las redes, FP32 frente a precisión mixta compatible. Compilación,
fusión y kernels propios solo se añaden ante un cuello de botella medido. Cada
alternativa mantiene las mismas muestras y registra desviación numérica y coste
total, incluido el inicio. Las mejoras de cómputo no se usan como prueba de mejor
predicción financiera.

## Recursos y recuperación

Hay una sola GPU. Se ejecuta una carga GPU principal a la vez y se solapan lectura
y preparación CPU mediante colas acotadas cuando la medida lo justifique. Los
hilos de Polars, BLAS y lectores comparten un presupuesto para evitar multiplicarlos.

Cada ejecución exige límites explícitos de RAM, VRAM, disco temporal y número de
trabajadores. El preflight combina capacidad física y memoria libre. Si no hay
margen para el perfil validado, la tarea espera o termina con un estado de falta
de recursos, conservando su checkpoint. No detiene otras aplicaciones ni cambia
silenciosamente a CPU. El objetivo es caudal útil sostenido, no llenar toda la RAM.

La continuación operativa es un prerrequisito de la campaña larga. El CLI debe
restaurar un directorio existente tras comprobar esquema, configuración, fuentes,
pesos, optimizador, precisión, RNG y cursor confirmado. Reanudar no crea otro ensayo
ni vuelve a aplicar muestras ya confirmadas. La prelectura no adelanta ese cursor.

Los checkpoints se publican de forma atómica en barreras coherentes. Los archivos
incompletos se ignoran, se conserva la retención acotada y se prueba una caída real
del proceso de ensayo. La recuperación interna de la sonda GRU es evidencia previa,
no sustituye ese contrato operativo.

## Evaluación, gráficos y cierre

Todos los modelos comparten manifiestos de muestras, particiones, objetivo y
reglas de transformación. Se registran configuraciones fallidas y descartadas.
Las semillas adicionales se aplican a modelos estocásticos, sin tratarlas como
mercados independientes. La búsqueda y la parada temprana usan validación cronológica.

Se conservan predicciones por activo y decisión, MAE por sesión, MSE, dirección y
Rank IC cuando el número de activos y los empates permitan calcularlo. Los intervalos
se obtienen sobre diferencias pareadas por bloques de sesiones. No se atribuye
calibración probabilística a una cabeza que solo emite un punto.

Los gráficos se generan desde resultados identificados: curvas de ajuste y
validación, error por ventana, cobertura por modalidad y periodo, tiempo por etapa,
caudal y picos de memoria. Los fallos y huecos permanecen visibles. Medidas con GPU
compartida se separan de perfiles sin interferencia. No se presenta una muestra
pequeña como estimación de p99 ni un sensor puntual como consumo energético total.

Una familia se considera terminada cuando procesó el 100 % de las muestras
admisibles de cada tramo asignado, terminó su política de ajuste, produjo las
predicciones previstas y pasó las comprobaciones de recuperación y correspondencia.
Una fuente sin resolver o una familia que no termina no se marca como completada.

Las tareas existentes coordinan el trabajo: #66 para cobertura y campaña, #50
para preparación por bloques, #67 para recuperación, #31 para ejecución y trazabilidad,
#24 a #27 para referencias, #52 para TCN y #32 para métricas. #95 corrige identidad
de gráficos y #94 amplía variables empresariales, sin reabrir ni duplicar entregas
ya completadas.

## Decisiones de implementación todavía condicionadas

El diseño no fija una duración de la campaña a partir de las 65 muestras. El
presupuesto se obtiene tras conocer la cobertura real y medir preparación,
representaciones, ajuste, validación, recuperación y escritura en esa carga.
Tampoco certifica que los 3.449 instrumentos con fuentes presentes produzcan muestras
válidas. Esa intersección debe resolverse por activo, decisión y disponibilidad.

Referencias técnicas: [ejecución streaming de Polars](https://docs.pola.rs/user-guide/concepts/streaming/),
[memoria externa de XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html),
[lectura de datos en PyTorch](https://docs.pytorch.org/docs/2.14/data.html) y
[regla delta de memoria rápida](https://proceedings.mlr.press/v139/schlag21a.html).
