# Puente de evidencia común en el catálogo de MARS-TITAN

`MarsTitanVariant` construye el puente desde el componente opcional `evidence_bridge` de `configs/titans/mars-titan-extensions.json`. La opción permanece apagada por defecto. Cada episodio conserva una sola identidad y un único registro en el banco. Las regiones tienen pesos y transformaciones propios sobre esa evidencia y comunican sus vistas mediante el módulo existente `SharedEvidenceBridge`.

El catálogo exige cinco opciones: `regions`, `topology`, `cycles`, `bridge_gain` y `seed`. Su validación es común a la configuración del puente y al constructor. Las regiones se copian como tupla y la ganancia se normaliza como número real. Cambiar cualquiera de las opciones cambia la identidad de la variante. El codec, dimensión, vecinos, temperatura y presupuestos proceden de la configuración del lector. No se aceptan como opciones regionales que puedan contradecirla.

El consumidor verifica la configuración del lector, la presencia y el tipo del módulo físico y la coincidencia entre ambos. También rechaza un puente físico oculto por metadatos que lo declaran ausente. La dependencia del banco episódico y la exclusión existente con B6 se conservan. Omitir el componente mantiene la identidad de configuración y variante anterior, incluido el núcleo con todo apagado. Las huellas de ejecución corresponden al código de cada revisión.

## Comprobaciones

La [regresión seleccionada](verification.json) pasó 239 casos y el complemento añadió ocho casos distintos, comprobados por sus identificadores. Son 247 casos, 78 marcados CUDA, sin fallos, errores ni omisiones. Después de ordenar las importaciones del nuevo archivo se repitieron sus 80 casos, 40 CUDA. Esa tanda se solapa con las anteriores y no se suma. No es toda la suite del proyecto.

Las pruebas nuevas comparan el constructor con configuraciones manuales para las tres topologías y K = 1, 2 y 4 en CPU y `cuda:0`. Comprueban identidad, normalización de valores numéricos, independencia de los datos del llamador, serialización de componentes, carga de `state_dict`, control M0 sin banco, RNG, opciones inválidas y lectores de otra combinación. Las sesiones nativas comparan M1, M2 y M3 con su construcción manual, con K = 4, episodios de la primera lectura y capacidad ocho. Coinciden exactamente las emisiones, identidad del contrato y modelo, snapshot, banco, cola pendiente y pesos rápidos comprobados.

Los seis casos de desacuerdo entre metadatos y módulo incluyen configuración física distinta, módulo ausente, tipo de metadatos incorrecto, forma declarada incompatible, tipo de configuración física incorrecto y una forma incompatible que ambos metadatos repiten. Otro caso intenta ocultar el módulo con metadatos ausentes. El catálogo los rechaza antes de crear la sesión.

Los datos son fixtures de cuatro identificadores con un historial de precios replicado, gráfico constante presente y noticias, fundamentales y macro ausentes con sus máscaras. El feedback es manual. M3 recibe escalas manuales identificadas como fixture, no estimadas con datos financieros. El padre y el lector están congelados. Las actualizaciones internas de memoria rápida siguen la regla de inferencia de Titans, sin pasos de optimizador externo.

La [mutación dirigida](mutations.json) detectó los 13 cambios seleccionados. Las referencias inicial y restaurada aprobaron 40 casos CPU cada una. Los mutantes omiten la construcción del puente, aceptan lectores incompatibles, ignoran metadatos físicos, borran opciones de la identidad, pierden los vecinos comunes, sustituyen K por uno, conservan alias del llamador, colapsan ganancia o semilla, retiran límites de ciclos, aceptan no finitos, ocultan un módulo o omiten la validación de formas compartidas. No es una puntuación sobre todas las mutaciones posibles.

La [cobertura y CRAP](quality.json) registra 334/342 sentencias y 125/134 ramas en los dos módulos de producción modificados. Se usaron coverage.py 7.16.1, Radon 6.0.1 y `scripts/summarize_python_quality.py`. Las tres funciones nuevas o ampliadas de conexión cubren todas sus sentencias y ramas en esta selección:

| Función | Sentencias | Ramas | CRAP |
| --- | ---: | ---: | ---: |
| `bridge_view_options` | 17/17 | 10/10 | 15 |
| `MarsTitanVariant._bridge_options` | 13/13 | 8/8 | 7 |
| `MarsTitanVariant.readout_config` | 9/9 | 6/6 | 5 |

CRAP utiliza cobertura de sentencias y publica las ramas por separado. Estas cifras no acreditan todas las combinaciones de operandos, formas o estados posibles.

Se conservan los intentos fallidos. El primer RED rechazó el componente no declarado. La primera tanda de construcción usaba una capacidad de cuatro con la frontera predeterminada de ocho. Se corrigió el fixture indicando su frontera y no se relajó la validación. La revisión adversarial reprodujo cuatro desacuerdos de metadatos antes de corregirlos. Un intento de cobertura seleccionando módulos por nombre falló durante la importación de PyTorch, antes de recoger pruebas. La selección por directorios permitió ejecutar el mismo conjunto y el resumen filtra los dos módulos modificados. El intento fallido no cuenta como cobertura.

## Datos y campaña

Los [recibos de la edición v3.1](../../../reports/data/edition-v3-1-verification-20261011.json) y de sus [vistas temporales](../../../reports/data/campaign-a-v2-views-v31-20261014.json) ya registran verificaciones completas del 11 y 14 de octubre. En este incremento se comprobaron las huellas actuales de los dos manifiestos y los tres informes de edición, vectores y objetivos. Coinciden con el recibo. También coincide la configuración de campaña. No se ha vuelto a recorrer todo el contenido Parquet ni a recalcular las codificaciones.

La [comprobación estática](readiness.json) de A v2 no enumera bloqueos de configuración ni familias pendientes. Su salida completa es idéntica antes y después del cambio de catálogo. No se ha añadido una variante a la campaña ni cambiado B. El alcance de ese comprobador es validar y contar la declaración. Los requisitos operativos de [#363](https://github.com/GonxKZ/mars-titan/issues/363), como coste y memoria sobre las vistas v3.1, regeneración CUDA y vinculación de recursos a una ejecución concreta, siguen separados de la preparación de datos ya verificada.

La conexión del catálogo no completa el colector ni el ajuste de alternativas regionales. Tampoco asigna sectores históricos a las regiones ni demuestra especialización aprendida. No se ejecutaron entrenamientos, postentrenamientos, RL ni evaluaciones científicas. El test final permanece cerrado. No hay mejora predictiva medida de esta variante.

Las [fuentes, XML, comprobaciones y registros externos](external-artifacts.json) fijan la versión de cada artefacto. El binario nativo se identifica por su SHA-256 y se reutilizó sin cambios de C++ o CUDA. No se ha medido una nueva latencia del constructor o del modelo completo, y este informe no atribuye aceleración a la conexión.

Con el entorno uv CUDA y el enlace nativo del proyecto, la selección nueva se ejecuta con:

```bash
uv run --no-sync pytest -q tests/memory/test_shared_bridge_variant.py
```

La orden incluye referencias CPU y casos explícitos de `cuda:0`. La evidencia publicada corresponde a las ejecuciones con sus XML y huellas, no a una nueva ejecución de esa orden en otro entorno.
