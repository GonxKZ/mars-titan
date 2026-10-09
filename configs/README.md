# Configuraciones de investigación

`study.toml` expresa decisiones iniciales del protocolo. Es una especificación de trabajo y no describe por sí sola toda la canalización implementada. Las fechas definitivas y el universo se fijarán tras la auditoría.

La selección principal propuesta es de hasta 128 activos, con piloto de hasta 64 y ampliación condicionada por tiempo. `checkpoints.toml` define la política que deberá implementar el futuro entrenador de MARS-TITAN para recuperar ejecuciones largas. No contiene ni genera pesos por sí mismo.

`baselines/historical-masked-campaign-a.json` y `historical-masked-campaign-b.json` declaran las dos variantes de presupuesto de la [campaña con máscaras desde 2000](../docs/research/training-campaign-2000.md#orquestación-de-los-brazos-con-entrenador). Comparten brazos, semillas y ventanas con la comparación walk-forward declarada y fijan el número máximo de ajustes y de predicciones trasladadas. Ninguna se ha ejecutado.

`titans/chronological-training-historical-masked.json` es la receta de los cuatro controles de Titans-MAC en esa campaña. Activa la memoria con residual y LayerNorm junto a `gate_bias` y declara dos casos de búsqueda de la tasa de aprendizaje. `chronological-training.json` y `chronological-training-quantile.json` son las recetas v1 y no cambian. Las tres se describen en el [entrenador cronológico](../docs/engineering/titans-chronological-trainer.md#receta-de-la-campaña-y-casos-de-búsqueda).

`titans/mars-titan-extensions.json` declara la variante MARS-TITAN con ampliaciones sobre Titans-MAC, con todos sus componentes apagados, cuáles están conectados, sus puntos de inserción y la matriz mínima de ablaciones. `titans/episodic-readout-historical-masked.json` es la receta del lector episódico sobre el Titans-MAC elegido, con los mismos dos casos de búsqueda y el presupuesto de Titans-MAC. No se han ejecutado. Su especificación está en la [arquitectura Titans-MAC](../docs/research/titans-mac-architecture.md#variante-mars-titan-con-ampliaciones).

`posttraining/` contiene la matriz de adaptadores declarada antes de cualquier ajuste. Cada futura ejecución conservará la configuración resuelta y su hash. Una configuración no podrá cambiar de significado según el cuaderno o directorio desde el que se ejecute. Las credenciales, si llegan a necesitarse para fuentes externas, quedarán fuera de los archivos versionados.
