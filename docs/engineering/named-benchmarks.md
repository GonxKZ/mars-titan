# Benchmarks identificados de memoria y adaptación

`mars_benchmarks_v1` da nombres y perfiles reproducibles a los ocho mecanismos de
[`adaptation-scenarios.json`](../../configs/simulation/adaptation-scenarios.json).
Reutiliza su generador, sus barras, su contexto y su almacenamiento. No cambia el
entorno ni la recompensa. La configuración original y sus resultados anteriores
siguen teniendo su propia identidad.

| Benchmark | Mecanismo existente | Qué permite contrastar |
| --- | --- | --- |
| Silencio | `no_signal` | Control negativo sin término de señal predictiva explícita. |
| Faro | `known_signal` | Control positivo con una señal disponible para el retorno siguiente. |
| Eco | `delayed_cue` | Retención de una pista que aparece una vez durante el calentamiento. |
| Giro | `signal_reversal` | Adaptación cuando se invierte la relación entre señal y retorno. |
| Retorno | `regime_recurrence` | Recuperación de una relación después de una secuencia A, B, A. |
| Contraste | `source_conflict` | Uso de fuentes contradictorias cuya fiabilidad está declarada. |
| Tormenta | `volatility_shift` | Respuesta a cambios de volatilidad, dispersión y saltos. |
| Umbral | `execution_friction` | Restricciones de volumen y operaciones corporativas ficticias. |

La palabra «control» describe el mecanismo. No asegura que un modelo alcance un
resultado concreto. Silencio conserva precios aleatorios y fricciones del entorno.
No es una demostración universal de que cualquier política tenga beneficio esperado
nulo. La verdad del generador sirve para comprobar los escenarios y permanece en
`evaluator/truth.parquet`, separada del contexto de la política.

## Perfiles y comparación

[`configs/benchmarks/scenarios.json`](../../configs/benchmarks/scenarios.json) fija
ocho activos por mundo y cinco semillas por familia y partición. Cada perfil tiene
120 mundos, repartidos entre `train`, `validation` y `audit`. Las semillas no se
repiten entre perfiles ni particiones.

El perfil `basic` usa 64 sesiones, con 16 de calentamiento y periodos de 16. El
perfil `extended` usa 256 sesiones, con 64 de calentamiento y periodos de 64. Ambos
incluyen tres periodos completos posteriores al calentamiento. En Eco, la pista
queda ocho sesiones antes de la primera decisión del perfil básico y 32 en el
prolongado. Este último exige conservar información durante más tiempo y procesa
más observaciones. La dificultad empírica no está medida y puede cambiar según el
algoritmo.

Cada comparación debe usar los mismos manifiestos, semillas, relojes y presupuestos
dentro de un perfil. Las filas de activos de una sesión no son réplicas temporales
independientes. Los dos perfiles no tienen igual exposición y no se deben mezclar
como si fueran una única comparación emparejada. La etiqueta `audit` reserva mundos
sintéticos al evaluador. No abre el test financiero final de 2024.

Estos escenarios tienen el contrato `technical_context_only`, con once campos de
contexto y tres conceptos macro simulados. No tienen las cuatro modalidades reales
de FinMultiTime ni sus 140 indicadores completos. El consumidor multimodal de
Titans necesita un adaptador propio y sus controles antes de utilizar otro contrato
de entrada. Tampoco se presentan estos casos como reproducciones de benchmarks
publicados. Las comparaciones históricas y las fuentes de algoritmos conservan sus
protocolos independientes.

## Preparación

```bash
uv run --locked python scripts/prepare_named_benchmarks.py --describe

uv run --locked python scripts/prepare_named_benchmarks.py \
  --profile basic --output data/interim/mars-benchmarks-v1-basic

uv run --locked python scripts/prepare_named_benchmarks.py \
  --profile extended --output data/interim/mars-benchmarks-v1-extended
```

El comando solo prepara datos. No ajusta un HMM, no contiene un bucle de optimizador
y no evalúa modelos. La implementación usa NumPy y PyArrow en CPU. Un perfil se
genera mundo a mundo y el catálogo completo admite como máximo 1.048.576 filas de
precios. Este límite acota el volumen previsto, no el RSS exacto del proceso.

`index.json` mantiene el formato del generador existente. `benchmarks.json` enlaza
su hash con los nombres, el perfil, la configuración, los hashes del código y las
versiones de Python, NumPy y PyArrow. Cambiar una receta o su implementación cambia
la identidad. Los lectores de escenarios conservan su formato, aunque una campaña
con un número de mundos fijado necesita otro protocolo para estos perfiles. En
particular, la campaña anterior de 896 mundos no admite este catálogo directamente.

Se rechazan identidades con rutas, claves desconocidas, familias duplicadas,
semillas solapadas, calendarios insuficientes y destinos existentes. Si falla la
generación, no se publica un manifiesto de benchmarks preparado. El índice parcial
identifica el fallo. La recuperación de esa generación no está implementada y
requiere un destino nuevo.

La preparación y las pruebas de contratos están permitidas durante el bloqueo de
aprendizaje. Entrenar, medir retención aprendida o comparar algoritmos queda pendiente
de completar y verificar la edición histórica desde 2000. No hay mejora predictiva
medida con este catálogo.

El [recibo técnico del 9 de octubre](../../reports/engineering/named-benchmarks-verification-20261009.json)
registra 56 pruebas, nueve mutaciones detectadas y la preparación de los 240 mundos.
La generación y verificación completa tardó 5,55 segundos dentro del proceso, en
una única ejecución con otras cargas presentes. Esa medida no acredita una
aceleración ni estima el coste de entrenar sobre los escenarios.

La [revisión independiente](../../reports/engineering/named-benchmarks-review-20261009.json)
añade ocho sondas y vuelve a comprobar las 56 pruebas seleccionadas. Contrasta las
ocho familias completas con el generador original, los extremos de semillas y
volumen, el RNG y la copia de configuración. No encontró defectos materiales en
ese alcance y no ejecutó ajustes ni evaluaciones de modelos.
