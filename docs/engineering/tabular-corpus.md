# Referencias tabulares sobre el corpus supervisado

Ridge y el control HistGradientBoosting consumen el mismo manifiesto y las mismas
etiquetas que las redes. Las ventanas de precios y los vectores de noticias,
gráficos, fundamentales y contexto macro se aplanan únicamente para el lote
actual. No hay un límite adicional de empresas ni una muestra reducida para
estas referencias.

Ridge utiliza la implementación float64 por bloques en `cuda:0`. Ajusta media,
escala e intercepto con entrenamiento y verifica que las dos pasadas usan los
mismos datos. No forma una matriz de ventanas de todo el corpus en RAM. El coste
de sus estadísticas es cuadrático en el número de características.

HistGradientBoosting es un control CPU con cuatro hilos, 30 iteraciones, siete
hojas máximas y sin reserva aleatoria de validación. Necesita una matriz completa
en memoria. El ejecutor estima su tamaño antes de ajustar. Si supera el presupuesto,
guarda `not_executed_budget` y cero filas ajustadas. No recorta filas para presentar
el caso como completado.

El límite predeterminado de matriz es de 256 MiB y puede declararse otro hasta
4 GiB. No es un límite del RSS total. La concatenación y las estructuras internas
del estimador necesitan memoria adicional. Una edición que no quepa requiere
medir otro presupuesto o incorporar una referencia de memoria externa identificada
como otro algoritmo. No se presupone equivalencia entre HistGradientBoosting y
XGBoost.

```bash
uv run --no-sync python -m mars_titan.training.tabular_corpus \
  --manifest data/processed/training-supervision-20260922/manifest.json \
  --output data/interim/tabular-edition-20260922/ridge-1 \
  --kind ridge --alpha 1
```

Para el control CPU se utiliza `--kind boosting` y una salida distinta. Cada
ejecución conserva configuración, versiones, huellas, recuentos, modelo y
predicciones en Parquet por bloques. Se comprueba que el modelo restaurado
predice exactamente igual. El RSS registrado es el máximo de vida del proceso,
no una reserva atribuible exclusivamente al último ajuste.

El ajuste tabular no tiene todavía reanudación interna de estadísticas o árboles.
Una interrupción deja su recibo y no se sobrescribe. Para repetir el ajuste se
necesita otra salida. Las predicciones terminadas conservan ambas particiones y
la referencia cero. El test final permanece cerrado.
