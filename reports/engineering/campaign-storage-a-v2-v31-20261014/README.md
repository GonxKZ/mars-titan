# Disco de la campaña A v2 con las vistas v3.1

Fecha: 14 de octubre de 2026. Tarea [#363](https://github.com/GonxKZ/mars-titan/issues/363).

Este informe repite la estimación de disco de la campaña A v2 con los recuentos reales de sus tres vistas sobre la edición v3.1 y la retención v2 recorrida ventana a ventana. No se ha ejecutado ningún entrenamiento, paso de optimizador ni evaluación científica, y no se ha leído ningún objetivo ni modelo. De las vistas solo se leen los manifiestos. Las cifras usan GB decimales (10⁹ bytes).

## Resumen

- Con la retención v2 declarada el 9 de octubre, que conserva compactadas las tablas de los adaptadores, A v2 conservaría 841,2 GB al terminar y su pico sería de 953,3 GB, aunque todas las regeneraciones salgan idénticas bit a bit. Si ninguna lo hace, serían 1.003,2 GB y 1.103,1 GB. El pico cae en la liberación de `fold-018`.
- Casi todo lo conservado son las tablas de los 10.404 ajustes de la etapa de adaptadores: entre 29 y 59 GB por ventana.
- Si las tablas de los adaptadores no se conservaran una vez escritos sus agregados por ventana, quedarían 18,2 GB al terminar con un pico de 55,4 GB con regeneraciones exactas, y 180,1 GB con un pico de 205,3 GB sin ellas.
- Después de preparar las tres vistas, el 14 de octubre a las 05:50, quedaban 17 GB libres en `/`. Con el margen declarado de 8 GiB, ni siquiera el mejor escenario cabe sin liberar espacio.

Las dos últimas cifras no corresponden a ninguna configuración declarada. Se calcularon con el mismo plan y el mismo recorrido, sustituyendo la parte de los adaptadores de cada ventana por cero (`rolling_storage.adapter_window`), para medir cuánto pesa esa decisión. La etapa de políticas depende de la cadena de adaptadores, así que no existe un plan de A v2 sin ellos.

## Entradas

| Ámbito | Ventanas | Ajuste | Validación | Calibración | Evaluación | Flujos máximos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| US+CN | 19 | 143.200.257 | 7.985.418 | 4.036.636 | 16.705.862 | 5.008 |
| US | 19 | 124.568.975 | 6.672.861 | 3.378.556 | 13.943.952 | 4.198 |
| CN | 13 | 18.046.778 | 1.103.090 | 541.055 | 2.218.863 | 810 |

Los recuentos de cada ventana están en [`counts.json`](counts.json). Salen de `masked_campaign.scope_views` y `campaign_storage.view_counts` sobre las vistas preparadas el 13 y el 14 de octubre, que `scope_views` valida al leerlas. Los bytes por fila y las medidas de las demás etapas son los de la [estimación del 9 de octubre](../campaign-storage-20261009/README.md): `row-bytes.json` y `extras-layout-640.json`.

El plan tiene 5.233 trabajos en la campaña base (2.207 elegidos), 5.415 predicciones de la ablación, 10.404 ajustes de adaptadores con 1.710 padres y 3.384 ajustes, 3.663 referencias y 3.663 evaluaciones de políticas sobre 3.960 cintas.

## Orden

```bash
uv run --no-sync python scripts/run_masked_campaign.py schedule \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json \
  --adapter-stage configs/posttraining/historical-masked-adapter-stage-a-v2.json \
  --ablation-stage configs/evaluation/historical-masked-ablation-stage-a-v2.json \
  --rl-stage configs/simulation/historical-masked-rl-stage-a-v2.json --output <orden>
uv run --no-sync python scripts/run_masked_campaign.py storage \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json \
  --storage configs/baselines/historical-masked-campaign-storage.json \
  --ablation-stage configs/evaluation/historical-masked-ablation-stage-a-v2.json \
  --adapter-stage configs/posttraining/historical-masked-adapter-stage-a-v2.json \
  --rl-stage configs/simulation/historical-masked-rl-stage-a-v2.json \
  --schedule <orden> --adapter-blocks \
  --extras reports/engineering/campaign-storage-20261009/extras-layout-640.json \
  --row-bytes reports/engineering/campaign-storage-20261009/row-bytes.json \
  --counts reports/engineering/campaign-storage-a-v2-v31-20261014/counts.json \
  --output <estimación>
```

La salida está en [`estimate.json`](estimate.json), con las rutas relativas al repositorio. Tarda unos 9 s y 0,8 GB de memoria.

## Por etapa

Sin recorrer por ventanas, cada etapa por separado conservaría:

| Etapa | Disposición actual | Grupos grandes | Tabla común |
| --- | ---: | ---: | ---: |
| Campaña base, con liberación al confirmar | 398,0 | 301,0 | 238,0 |
| Ablación de modalidades | 311,1 | 203,3 | 135,0 |
| Adaptadores | 1.777,9 | 1.227,3 | 808,7 |
| Cintas de las políticas | 6,7 | | |

El recorrido por ventanas reduce mucho la base y la ablación, porque libera sus filas regeneradas y sus índices, pero no los adaptadores, que la retención v2 compacta y conserva.

## Dos correcciones previas

Repetir la estimación sacó a la luz dos fallos, corregidos junto a este informe:

1. `extras-layout-640.json`, la medida que exigen las órdenes `storage` y `rolling` con la etapa de políticas, no tenía `comparison_aggregate_bytes_per_series`, que la publicación de [#541](https://github.com/GonxKZ/mars-titan/pull/541) añadió solo a `extras.json`. Con la etapa de adaptadores, la estimación y la guardia por ventana de `rolling` terminaban con un `KeyError`. La prueba de la medida publicada exige ahora que las dos coincidan salvo en las cintas.
2. El recorrido por ventanas sumaba para siempre lo que conserva cada meseta de la parada conjunta: sus estados de recuperación y, en la familia Titans, sus índices de observaciones. La continuación reanuda en la carpeta de la meseta y `release_confirmed` la libera al confirmarse, así que de la meseta solo queda la copia de su informe. Con el cálculo anterior la estimación daba 994,7 GB conservados y 1.106,7 GB de pico, unos 153 GB de más, y la guardia de `rolling` habría proyectado crecimientos que no ocurren. El pico de cada ventana sigue contando a la vez los índices de todas las mesetas que esperan su época común.

## Qué queda por decidir

- El destino de las tablas de los adaptadores. Liberarlas exige una regeneración por inferencia de cada adaptador, que no existe, o aceptar que se borren con su huella y sus agregados ya escritos, sin poder repetirlas. Cualquiera de las dos cambia la retención v2 declarada y debe registrarse como modificación del protocolo antes de ejecutar.
- El espacio libre. Incluso sin conservar los adaptadores, el mejor escenario necesita unos 64 GB libres al empezar (55,4 GB de pico más el margen), y había 17 GB.
- Comprobar en `cuda:0` que la regeneración de cada familia repite sus tablas bit a bit, porque de ello depende que el resultado se acerque a 18 GB o a 180 GB.
