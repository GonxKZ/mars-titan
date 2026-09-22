# Configuraciones de investigación

`study.toml` expresa decisiones iniciales del protocolo. Es una especificación de trabajo y no describe por sí sola toda la canalización implementada. Las fechas definitivas y el universo se fijarán tras la auditoría.

La selección principal propuesta es de hasta 128 activos, con piloto de hasta 64 y ampliación condicionada por tiempo. `checkpoints.toml` define la política que deberá implementar el futuro entrenador de MARS-TITAN para recuperar ejecuciones largas. No contiene ni genera pesos por sí mismo.

Cada futura ejecución conservará la configuración resuelta y su hash. Una configuración no podrá cambiar de significado según el cuaderno o directorio desde el que se ejecute. Las credenciales, si llegan a necesitarse para fuentes externas, quedarán fuera de los archivos versionados.
