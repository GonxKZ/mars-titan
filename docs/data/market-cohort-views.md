# Vistas completas por mercado

Una comparación estadounidense no necesita esperar a que otro mercado resuelva
sus fuentes. La selección se hace sobre una preparación terminada e incluye
todos los candidatos del mercado elegido, tanto los preparados como los que
carecen de alguna modalidad. No admite fallos pendientes dentro de esa selección.

La vista es otro manifiesto. No copia ni modifica los datos originales o sus
derivados. Conserva la huella del origen, su estado, sus recuentos, sus fallos
y los mercados excluidos. Esos campos acompañan a las representaciones y a la
supervisión. Una vista completa de US no se presenta como un corpus completo
de US y CN.

El origen debe haber terminado su recorrido y reconciliar todos sus candidatos.
Se rechazan duplicados, cambios de manifiestos preparados y destinos con otra
identidad. No se encadenan vistas, para mantener directamente visibles los
límites de la preparación original.

```bash
uv run --no-sync python -m mars_titan.data.cohort_views \
  --source data/processed/original-audited-20260923/manifest.json \
  --output data/processed/original-audited-us-20260923/preparation.json \
  --market US
```

La vista estadounidense creada contiene 4.784 candidatos. Hay 2.639 activos
preparados y 2.145 sin todas las fuentes. De los preparados, 2.234 tienen precios,
noticias y hechos contables no vacíos. Ese último recuento es anterior a comprobar
la coincidencia temporal de las cuatro modalidades, los indicadores macro y
las etiquetas. No es todavía el número de empresas de un entrenamiento.

La recuperación china posterior terminó con 892 candidatos, 810 preparados y
82 sin todas las fuentes, sin fallos de proceso. Conserva 2.891.924 precios y
553.392 noticias. Ninguno de esos 810 activos tiene hechos contables admitidos
por la preparación actual, al faltar divulgación acreditada. Esto impide iniciar
la comparación china y la mixta con cuatro modalidades completas. Las cifras
no se fechan a partir del cierre del periodo para eludir esa limitación.

La implementación pasó 1.132 pruebas locales, sin omisiones, incluidas CUDA y
las comprobaciones de diagnóstico nativo. Las pruebas específicas contrastan
la conservación de la población, los errores del origen, la recuperación sin
reescritura y la propagación del alcance a las etiquetas.

La [cobertura y los diagnósticos](../../reports/resources/market-cohort-views-quality.json)
corresponden a las pruebas específicas. Dos mutaciones dirigidas confirmaron
que esas pruebas detectan la pérdida de candidatos incompletos y la aceptación
de un mercado seleccionado que todavía contiene fallos.
