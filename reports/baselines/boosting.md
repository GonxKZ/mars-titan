# Referencia de boosting tabular

Se utiliza `HistGradientBoostingRegressor` de scikit-learn como referencia CPU
explícita. No es una sustitución por falta de CUDA. La sonda conserva las mismas
cuatro modalidades, macro, etiquetas y cortes temporales que Ridge.

La configuración inicial usa 30 iteraciones, hasta siete hojas, un mínimo de cinco
muestras por hoja, tasa 0,05 y semilla 42. `early_stopping=False` impide que el
estimador reserve aleatoriamente parte del panel para validación. No se ha hecho
búsqueda de configuraciones con estas quince filas de validación.

## Memoria y persistencia

La matriz se recibe por bloques y tiene un presupuesto de 256 MiB para sus valores.
Cada bloque se copia para que un lector que reutiliza buffers no modifique filas
anteriores. El ajuste del estimador sigue siendo en memoria, no incremental.
Durante la concatenación coexisten bloques y matriz, y el estimador añade sus
propias estructuras. Por tanto, 256 MiB no es un límite del RSS total. Los hilos
de las bibliotecas numéricas se acotan a cuatro.

Se guarda un artefacto local mediante joblib, con publicación atómica y sin
sobrescribir una ejecución anterior. La recarga exige la huella conservada de
ese mismo artefacto. Joblib no se usa para modelos descargados ni archivos de
terceros. Una huella correcta no convierte un archivo malicioso en seguro.

## Ejecución observada

La [sonda CPU](../resources/strict-boosting-probe.json) ajustó las 50 muestras
anteriores a 2023 y evaluó las mismas 15 de 2023 que Ridge. Usó 1.660 características.
El ajuste tardó 0,789 segundos y el recorrido completo, 2,254 segundos. El pico
RSS del proceso fue de 734,31 MiB, incluidas las bibliotecas y la preparación de
etiquetas. No reservó VRAM.

El MAE diagnóstico fue 0,011771, frente a 0,009620 del predictor cero. Este resultado
también es peor que cero. Se conserva sin ajustar la muestra ni repetir configuraciones
hasta obtener una mejora. Su tamaño y selección dirigida no permiten una conclusión
confirmatoria sobre capacidad predictiva.

Las predicciones recargadas fueron idénticas. El informe registra parámetros,
fuentes y artefactos. Las versiones completas se reproducen con `uv.lock`, aunque
el recibo aún no incluye directamente las versiones de scikit-learn y joblib.

```bash
uv run --locked --extra research --extra cuda --extra encoders \
  python -m mars_titan.reference_probe --kind boosting \
  --prepared data/processed/news-expansion-20260921 \
  --samples data/processed/bounded-samples-20260921-v2/US \
  --output data/interim/boosting-probe-new \
  --report reports/resources/boosting-probe-new.json
```

Los extras mantienen el entorno común de lectura y etiquetas. La ruta de boosting
no llama a CUDA y las pruebas lo comprueban expresamente.

## Verificación y trabajo pendiente

La suite sin GPU pasó con 422 pruebas y 16 omitidas por necesitar CUDA o activación
explícita. Se comprobaron todas las filas de ajuste, ausencia de validación aleatoria,
buffers reutilizados, presupuesto de matriz, valores no finitos y persistencia.
La revisión independiente reprodujo las predicciones y métricas del informe.

#25 sigue abierta para la evaluación por ventanas y una cohorte suficiente. No se
incorpora memoria externa ni otra biblioteca de boosting sin una necesidad medida.
