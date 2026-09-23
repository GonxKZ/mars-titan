# Campaña tabular sobre la población común

La búsqueda compara Ridge y XGBoost con las mismas muestras, etiquetas y cortes
que las referencias neuronales. No reduce filas para acomodar una matriz en
memoria. Las cuatro modalidades y macro se leen desde los Parquet confirmados.

La [configuración estadounidense](../../configs/baselines/tabular-search-us.json)
define tres regularizaciones Ridge, 0,1, 1 y 10. XGBoost compara doce combinaciones
de profundidad 3 o 6, 64, 128 o 256 bins y tasa 0,03 o 0,1. Cada ajuste tiene
200 rondas. El menor MAE por sesión en validación selecciona el finalista, que
se repite con semillas 43 y 44 además de la 42 de búsqueda. Son 17 ajustes.
Los empates conservan el identificador menor. Las semillas no son observaciones
financieras independientes y pueden producir resultados iguales en configuraciones
deterministas.

Se usa `cuda:0` de forma explícita. La caché de XGBoost va a disco con
`on_host=false`. El lote es de 1.024 filas, el presupuesto por bloque es de
64 MiB y el límite configurado de caché de host es de 2 GiB. Estos parámetros
no equivalen a un límite absoluto de toda la RAM o VRAM del proceso. Se registra
el uso observado y se mantiene un límite externo para el proceso de la campaña.

## Confirmación y recuperación

Cada caso conserva sus parámetros, intentos, huellas, checkpoint y predicciones.
El resumen confirma un resultado solo si coinciden origen, población, modelo,
parámetros y número de rondas, y si sus artefactos mantienen sus hashes.
Los resultados terminados no se vuelven a entrenar al reanudar.

Ridge no tiene un checkpoint de estadísticas parciales. Si su ajuste se interrumpe,
el intento permanece y se empieza otro en un directorio nuevo. Una interrupción
después de publicar su informe completo permite reutilizarlo aunque todavía no
se hubiese confirmado en el resumen de la campaña.

XGBoost recupera el modelo confirmado y continúa las rondas pendientes. Se solicita
una confirmación cada diez rondas. Durante la preparación también se comprueba
la petición de parada. No se garantiza que una interrupción sea atendida después
de cada ronda individual. Una terminación forzada puede dejar una caché temporal
en disco. No se borran esos archivos ni se encadenan reintentos automáticamente.

La inicialización tiene una identidad propia. Un fallo al escribir el primer
resumen puede retomarse si todavía no existen modelos ni otros artefactos.
Una carpeta con archivos ajenos o la desaparición de un resumen posterior no
se interpreta como permiso para empezar desde cero. Un bloqueo impide dos
escritores simultáneos. Los límites de semillas y rondas se comprueban antes
de entrenar el primer modelo, y los identificadores no redondean parámetros
distintos al mismo nombre.

## Dependencia de la campaña neuronal

`mars_titan.training.baseline_queue` comprueba que la búsqueda neuronal esté
terminada, con todos sus informes, checkpoints y predicciones íntegros. Después
recupera la vista exacta del brazo con ponderación natural. Esa comprobación
se conserva en un recibo separado y su hash se exige al iniciar el estudio
tabular. Una campaña neuronal pausada o fallida no activa los ajustes siguientes.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync \
  --with xgboost==3.3.0 --with cupy-cuda13x==14.2.0 \
  python -m mars_titan.training.baseline_queue \
  --config configs/baselines/tabular-search-us.json \
  --reference data/interim/original-audited-us-scientific-search-20260923/summary.json \
  --output data/interim/original-audited-us-tabular-search-20260923
```

El módulo `tabular_search` también acepta directamente `--config`, `--manifest`,
`--output` y `--resume`. El orden secuencial entre familias evita competir por
la misma GPU, sin necesitar un servicio distribuido. La ejecución y los datos
permanecen locales.

## Evidencia y límites

La [verificación local](../../reports/resources/tabular-search-quality.json)
registra 49 pruebas del estudio, la cola y los entrenadores que reutiliza.
Incluye un Ridge y dos XGBoost reales en CUDA sobre una población de prueba,
además de cuatro mutaciones dirigidas detectadas. Los tres problemas de la
revisión independiente tienen regresiones específicas. No se ha repetido la
batería completa durante la campaña neuronal que ocupa la GPU.

Esta comprobación no acredita que los 17 ajustes amplios hayan terminado.
La reserva final permanece cerrada. La adaptación predictiva se ejecutará
sobre padres confirmados y seguirá separada de esta selección tabular.
