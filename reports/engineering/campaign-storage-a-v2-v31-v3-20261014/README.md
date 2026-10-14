# Disco de la campaña A v2 con la retención v3

Fecha: 14 de octubre de 2026. Tareas [#611](https://github.com/GonxKZ/mars-titan/issues/611) y [#363](https://github.com/GonxKZ/mars-titan/issues/363).

Este informe repite la [estimación anterior](../campaign-storage-a-v2-v31-20261014/README.md) con la retención v3, declarada el 14 de octubre en [`historical-masked-retention-v3.json`](../../../configs/baselines/historical-masked-retention-v3.json) antes de cualquier resultado. El plan, los recuentos de las vistas v3.1, las medidas y el orden por ventanas son los mismos. La etapa de adaptadores no cambia: siguen los 10.404 ajustes con sus variantes, semillas, reglas de selección y presupuestos. No se ha ejecutado ningún entrenamiento, paso de optimizador ni evaluación científica. Las cifras usan GB decimales (10⁹ bytes).

## Qué cambia la v3

La v2 compactaba las tablas por fila de los adaptadores y las conservaba, y eso suponía 809 GB. La v3 las libera con la misma garantía que ya tenían la base y la ablación:

- Cada ajuste de adaptadores compacta sus tablas sin pérdida al confirmarse y conserva solo su estado elegido. Los estados de recuperación se borran después de confirmar el recibo.
- Al cerrar un bloque (un modelo base en una ventana, con todas sus semillas y variantes), se escriben desde las filas los agregados de la comparación postentrenada. Después, cada tabla se regenera por inferencia desde el estado elegido, con FP32 estricto, y se compara bit a bit con la huella de su recibo. Solo entonces se libera.
- Las evaluaciones del predictor de la cadena que leen las políticas de ventanas posteriores se conservan compactadas hasta su último lector.
- Si una regeneración de adaptadores no sale idéntica, sus tablas quedan compactadas y el recorrido se detiene, también al reanudarlo, hasta que se revise. Seguir conservando tablas compactadas llevaría al escenario sin regeneraciones, que no cabe.
- La caché de cada padre congelado se borra al cerrarlo. La regeneración la vuelve a calcular.
- La base y la ablación compactan sus tablas al terminar su fase dentro de la ventana, en lugar de esperar a la liberación.

La v2 sigue declarada y sin cambios. La v3 copia todos sus campos, cambia `release` y `compact` y añade cuatro (`adapter_states`, `adapter_release`, `adapter_mismatch` y `parent_caches`). Su huella es `241bf69ee9f8b6801b723049269dcc24c5c2d86a0593cd8f14615abdbfb0175a`.

## Resultado

| Retención y escenario | Conservado | Pico | Momento del pico |
| --- | ---: | ---: | --- |
| v2, todas las regeneraciones idénticas | 841,2 | 953,3 | `fold-018/release` |
| v3, todas las regeneraciones idénticas | 54,7 | 92,6 | `fold-018/ablation` |
| v3, la base y la ablación no se repiten y los adaptadores sí | 216,3 | 242,1 | `fold-018/ablation` |
| v3, ninguna regeneración se repite | 850,8 | 863,1 | `fold-018/ablation` |

El último escenario es una cota que la v3 no alcanza, porque se detiene en el primer bloque de adaptadores que no se repite. La guardia de cada ventana proyecta un crecimiento máximo de 30,5 GB (`fold-017`) en los dos primeros escenarios de la v3.

A las 13:17 del 14 de octubre `/` tenía 131,1 GB libres. Con el margen declarado de 8 GiB quedan 122,5 GB, así que el escenario con todas las regeneraciones idénticas cabe con unos 30 GB de holgura. Si la base o la ablación no se repitieran en `cuda:0`, sus tablas compactadas se conservarían y la guardia detendría el recorrido antes de `fold-010`, con unos 97,5 GB conservados. Por eso comprobar esas regeneraciones en `cuda:0` es el paso previo al lanzamiento.

Las cifras de cada ventana y escenario están en [`estimate.json`](estimate.json). Fuera de la sección `rolling`, el archivo es idéntico al de la estimación anterior.

## Qué se conserva para comparar después

- **Comparación postentrenada.** Sus agregados se escriben desde las filas de cada bloque antes de liberarlas, con la misma identidad que la comparación que lee filas. La prueba de extremo a extremo comprueba que la publicación desde los agregados coincide con la calculada antes de liberar.
- **Tablas recuperables.** `run_masked_campaign.py regenerate` con `--adapter-stage` y `--adapter-output` vuelve a escribir cualquier tabla desde su estado elegido y la compara con la huella del recibo. Una métrica que no esté en los agregados se puede calcular después sobre filas idénticas a las originales, con el coste de una pasada de inferencia.
- **Selección de la cadena y disyunción.** Leen recibos, selecciones y etiquetas, no tablas, así que no cambian.

## Comprobaciones

Pruebas en CPU con la campaña reducida, sin pasos de optimizador:

- Las cinco familias de trabajos de la etapa (padre congelado, ajuste cronológico de Titans-MAC, ajustes de matriz, lectores de MARS-TITAN y adaptadores de la GRU candidata) regeneran sus tablas bit a bit desde el estado elegido. Con un peso alterado, la regeneración deja de ser idéntica.
- [`test_retention_v3.py`](../../../tests/posttraining/test_retention_v3.py) recorre la etapa dentro de la retención v3: compacta al confirmar, deja un estado por ajuste y ninguna caché, libera los siete trabajos por bloques con regeneraciones idénticas y publica igual que desde las filas. Otra prueba altera una regeneración y comprueba que sus tablas quedan compactadas y que el recorrido se detiene, también al reanudar.
- `test_rolling_storage.py` comprueba la contabilidad de la v3 en el estimador y `test_rolling_retention.py` que su declaración no admite cambios.

Quedan sin comprobar:

- La regeneración bit a bit en `cuda:0` con estados reales. Exige levantar el bloqueo de aprendizaje y la GPU libre.
- El coste de las regeneraciones. Cada uno de los 10.404 ajustes añade una pasada de inferencia sobre validación, calibración y evaluación, y cada uno de los 1.710 padres vuelve a calcular su caché. No está medido. Se medirá en la primera ventana.

## Orden

```bash
uv run --no-sync python scripts/run_masked_campaign.py storage \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json \
  --storage configs/baselines/historical-masked-campaign-storage.json \
  --ablation-stage configs/evaluation/historical-masked-ablation-stage-a-v2.json \
  --adapter-stage configs/posttraining/historical-masked-adapter-stage-a-v2.json \
  --rl-stage configs/simulation/historical-masked-rl-stage-a-v2.json \
  --schedule <orden> --adapter-blocks \
  --retention configs/baselines/historical-masked-retention-v3.json \
  --extras reports/engineering/campaign-storage-20261009/extras-layout-640.json \
  --row-bytes reports/engineering/campaign-storage-20261009/row-bytes.json \
  --counts reports/engineering/campaign-storage-a-v2-v31-20261014/counts.json \
  --output <estimación>
```

Tardó 27 s y 0,8 GB de memoria, con otras pruebas en marcha. Sin `--retention` la orden estima la v2, como antes.
