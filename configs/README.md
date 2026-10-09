# Configuraciones de investigación

`study.toml` expresa decisiones iniciales del protocolo. Es una especificación de trabajo y no describe por sí sola toda la canalización implementada. Las fechas definitivas y el universo se fijarán tras la auditoría.

La selección principal propuesta es de hasta 128 activos, con piloto de hasta 64 y ampliación condicionada por tiempo. `checkpoints.toml` define la política que deberá implementar el futuro entrenador de MARS-TITAN para recuperar ejecuciones largas. No contiene ni genera pesos por sí mismo.

`baselines/historical-masked-campaign-a.json` y `historical-masked-campaign-b.json` declaran las dos variantes de presupuesto de la [campaña con máscaras desde 2000](../docs/research/training-campaign-2000.md#orquestación-de-los-brazos-con-entrenador). Comparten brazos, semillas y ventanas con la comparación walk-forward declarada y fijan el número máximo de ajustes y de predicciones trasladadas. Ninguna se ha ejecutado.

`posttraining/` contiene la matriz de adaptadores declarada antes de cualquier ajuste. Cada futura ejecución conservará la configuración resuelta y su hash. Una configuración no podrá cambiar de significado según el cuaderno o directorio desde el que se ejecute. Las credenciales, si llegan a necesitarse para fuentes externas, quedarán fuera de los archivos versionados.
