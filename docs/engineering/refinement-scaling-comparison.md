# Comparación de las cohortes de refinamiento

`benchmarks.refinement_scaling_analysis` reúne cuatro estudios: 64 y 256
activos, cada uno con la representación gráfica original y la geométrica. El
análisis reutiliza las evaluaciones confirmadas de las rutas K1 y K4cc y de la
política seleccionada. No cambia selecciones, modelos ni predicciones.

Hay tres poblaciones de comparación. Las dos primeras conservan todos los
activos de cada tamaño y contrastan original con geometría. La tercera contiene
los 64 activos comunes fijados antes del ajuste y compara las cuatro
combinaciones sobre las mismas filas y objetivos. Una fila ausente, repetida o
con otro objetivo invalida el contraste. El análisis no calcula una intersección
de filas disponibles para continuar.

Para cada ruta, la familia sobre los activos comunes incluye geometría menos
original en cada tamaño, 256 menos 64 en cada representación y la interacción:

```text
I = MAE_256_geometría − MAE_256_original − MAE_64_geometría + MAE_64_original
```

También compara la política con K1, K4cc y retorno residual cero dentro de cada
combinación. Son 27 contrastes en la familia común y nueve en cada población
completa. Un delta negativo indica menor MAE de la variante respecto a la base. La
mejora relativa conserva la convención del motor estadístico, positiva cuando
disminuye el error y no definida si el error de la base es cero.

El MAE se calcula primero por sesión. El bootstrap circular remuestrea las fechas
UTC observadas, con todos sus activos juntos, usando 2.000 réplicas, semilla
20261018, bloques de 16 fechas y sensibilidad con 8 y 32. Los días sin observaciones
no ocupan una posición del índice. Los intervalos simultáneos se calculan dentro
de cada familia. Esa corrección no cubre conjuntamente todas las proyecciones,
lectores y semillas del estudio.

Cada proyección, lector y semilla se conserva por separado. La media de pérdidas
por sesión solo se publica cuando están completas las tres semillas previstas,
42, 43 y 44. No añade fechas ni produce una predicción de ensemble. Los
contrastes individuales por semilla permanecen en el resultado. Si falta una
combinación, se identifica como pendiente y no aporta filas a ese grupo.

Las preparaciones deben conservar código, entorno, configuración y catálogo de
trabajos, salvo el tamaño de cohorte previsto. Se verifican ambos sellos de
selección, la vinculación de cada fase a su preparación, las huellas de los
Parquet y el recuento de filas. La CLI exige una salida nueva fuera de las
fuentes y respeta la protección local de aprendizaje. Cada lectura de Parquet
está limitada a 131.072 filas y 64 MiB. Los cálculos estadísticos usan las
implementaciones CPU existentes.

La salida incluye contrastes por semilla, medias completas, grupos pendientes y
tablas de MAE por sesión. `utc_day_index` es la posición, empezando en cero, de
cada fecha UTC en la lista ordenada de fechas observadas de esa población.
Un viernes y el lunes siguiente tienen índices consecutivos si no hay datos del
fin de semana. Las huellas enlazan estas tablas con las
predicciones originales, que conservan sus fechas absolutas.

Ampliar la cohorte cambia los datos de ajuste y la composición del banco
episódico. La comparación sobre los 64 comunes controla las filas puntuadas,
pero no separa esos dos cambios. Tampoco convierte el desarrollo reutilizado
de 2023 en una evaluación final. Los resultados completos de cada estudio
conservan sus demás controles y métricas.

La CLI acepta `--n64-original`, `--n64-geometry`, `--n256-original`,
`--n256-geometry` y `--output`. La
[verificación inicial](../../reports/engineering/refinement-scaling-analysis-20261019/README.md)
cubre fixtures y el contraste de metadatos de las cuatro preparaciones. El
[resultado completo](../../reports/research/refinement-development-20261019/completed-v1/README.md)
reúne las tres semillas, las medias de pérdidas por sesión y el cierre de las
288 tareas. Ninguno de los 48 selectores supera al control residual cero.
