# Rutas acotadas de consulta y refinamiento

La ampliación de [#56](https://github.com/GonxKZ/mars-titan/issues/56) ejecuta hasta cuatro
refinamientos del lector episódico sobre una sola preparación de Titans-MAC. Conserva los
pesos del núcleo, sus actualizaciones asociativas, la cabeza y el snapshot de memoria.
Su implementación para estudios de desarrollo está separada de la campaña y no cambia
los modelos ni los valores por defecto existentes.

Después de K1, C atiende los episodios ya seleccionados con la nueva consulta. Reutiliza
sus claves y valores proyectados. R selecciona vecinos de nuevo antes de refinar. El
bloque final contiene dos refinamientos y no publica una salida K3.

| Ruta | Refinamientos | Búsquedas con banco no vacío | Selección posterior a K1 |
| --- | ---: | ---: | --- |
| `k1` | 1 | 1 | Terminar |
| `k2c` | 2 | 1 | Conservar episodios |
| `k2r` | 2 | 2 | Renovar episodios |
| `k4cc` | 4 | 1 | Conservar en ambos bloques |
| `k4cr` | 4 | 2 | Conservar en K2, renovar antes de K3 |
| `k4rc` | 4 | 2 | Renovar antes de K2, conservar después |

La ruta anterior `per_step` sigue siendo otro control. Puede buscar en cada paso y no se
identifica con `k4rc` ni `k4cr`. El banco vacío y M0 ejecutan el refinador, pero no realizan
búsquedas ni proyecciones de valores. `value_projections` cuenta estas últimas, no todas
las capas lineales del modelo.

## Cálculo y coste

[refinement_routes.py](../../src/mars_titan/models/titans/refinement_routes.py) ejecuta una
ruta o recoge las seis compartiendo prefijos. La colección necesita nueve refinamientos
y tres búsquedas cuando hay episodios. Su coste pertenece a la producción de trazas de
desarrollo. No representa el coste de desplegar una política que devuelve una sola salida.

La caché contiene únicamente los episodios seleccionados y sus proyecciones. Vive dentro
de la decisión y se descarta después. La atención se recalcula con el estado actual. No
se conserva una caché entre cambios de parámetros, snapshots o predicciones. El presupuesto
de trabajo usa los pasos realmente ejecutados, incluido el total de la colección, y se
comprueba antes de comenzar. Es una estimación conservadora del código, no una medida
del pico de memoria del proceso.

[refinement_policy.py](../../src/mars_titan/models/titans/refinement_policy.py) implementa
primero una decisión única después de K1. Agrupa las filas por ruta y ejecuta sus
continuaciones. Las filas que paran quedan fuera de los bloques siguientes. Los códigos
producidos en CUDA necesitan una copia explícita a CPU para planificar los grupos.
Una tupla de códigos ya disponible en CPU evita ese viaje. La sincronización y el
cambio de tamaño de las matrices forman parte de la medida de latencia y de la
comprobación numérica. Reducir pasos medios no garantiza reducir p95 o p99.

Los rasgos de la puerta son la predicción inicial y la de K1, su diferencia, la anchura
del intervalo bruto cuando existe, la entropía y el peso máximo de atención, las normas
cuadráticas medias de la lectura y del cambio de estado y la presencia de memoria.
No contienen etiquetas ni resultados de las rutas todavía no ejecutadas. La anchura
bruta, la entropía y el cambio latente no se interpretan como confianza calibrada.

## Objetivo de la política

Para cada continuación r se conserva la ganancia firmada

\[
g_r = |y-\widehat y_{K1}|-|y-\widehat y_r|.
\]

Una ganancia negativa significa que continuar perjudicó esa observación.
[refinement_policy_fit.py](../../src/mars_titan/training/refinement_policy_fit.py) ajusta
cinco regresiones ridge con pesos por sesión y normalización calculada solo en ajuste.
También regulariza el intercepto hacia cero. No convierte la rama retrospectivamente
mejor en una etiqueta de clasificación ni trunca las ganancias negativas.

La política escoge el máximo entre cero, correspondiente a K1, y las cinco ganancias
previstas menos sus costes y un margen mínimo. Los costes son incrementos respecto a K1
expresados en unidades de MAE mediante una conversión declarada por el protocolo. No son
milisegundos sin convertir. Los empates siguen el orden de la tabla. Los límites son
rangos marginales por rasgo. Pertenecer a ellos no acredita soporte conjunto ni una
combinación observada. Fuera de esos rangos devuelve K1. Esta regla no es un detector
OOD calibrado.

El [alcance matemático de la política](refinement-policy-mathematics.md) define la
utilidad penalizada y una cota condicional de regret. La cota requiere un error
uniforme que el ridge actual no certifica y no cubre el retorno forzado a K1 fuera
de los rangos marginales.

El ajuste rechaza filas que precedan a la selección del núcleo y lector, decisiones fuera
del intervalo y etiquetas que maduren después de su frontera. Esas comprobaciones
dependen de los tiempos aportados. Los recibos de los checkpoints, las vistas y las
trazas deben acreditar su procedencia. Una fecha declarada no sustituye esa evidencia.

## Recorrido temporal y controles

[refinement_inference.py](../../src/mars_titan/training/refinement_inference.py) reutiliza
el recorrido cronológico de `MarsTitanInference`. Solo admite M0 o M1 para este estudio.
M1 conserva un banco que no depende del error de la ruta elegida. No se extiende esa
equivalencia a M2 o M3. Cada fase reinicia sus estados y utiliza el calentamiento de
entradas pasadas de la receta existente.

La colección guarda rasgos y predicciones alternativas al decidir. Los une con la
etiqueta en el evento de maduración y entonces publica la traza. El buffer pendiente
está limitado a 32.768 filas. Una fase interrumpida no autoriza a reutilizar un buffer
parcial como si estuviera completo. El estudio debe publicar fases y recibos de forma
atómica y recuperar desde un límite confirmado.

El diseño temporal previsto utiliza un núcleo y lector seleccionados en la ventana
anterior. Para evaluar en 2023, ajusta la puerta de enero a marzo de 2022, selecciona
de abril a septiembre y calibra la política completa de octubre a diciembre. Reutiliza
`staged_window` y `fit_rows_proof` para comprobar la disjunción. La evaluación de 2023
es exploratoria. El test final de 2024 sigue reservado.

La comparación necesita K fijos, `first_read`, `per_step`, una regla de coste sencilla,
una asignación aleatoria con presupuesto comparable, M0 y el retorno residual cero.
Debe separar utilidad de consultar, profundidad adicional y contracción de la salida
hacia cero. Se mide la política realmente ejecutada, con MAE por sesión, incertidumbre
temporal por bloques y coste completo. Los parámetros de la puerta quedan congelados
durante evaluación y la calibración no selecciona arquitectura.

[refinement_controls.py](../../src/mars_titan/models/titans/refinement_controls.py)
añade dos controles desplegables. `ThresholdRoutePolicy` elige una continuación declarada
si un rasgo de K1 supera estrictamente su umbral, comparado en FP64. Es una regla sencilla,
sin pesos aprendidos ni estimación de ganancia. `HashedRoutes` asigna rutas de forma
pseudoaleatoria determinista mediante SHA-256 de semilla, flujo y fecha. Usa los primeros
64 bits y una CDF con límites enteros. Sus seis recuentos int64 no negativos se fijan
con las frecuencias de validación, nunca con evaluación. Conserva la elección al permutar
o dividir lotes, pero no garantiza recuentos exactos ni el mismo coste realizado en una
muestra finita. El consumidor identifica ambos controles y registra sus rutas efectivas sin consultar
etiquetas ni avanzar MAC más veces.

Esta implementación no es evidencia de mejora predictiva ni de novedad del mecanismo
general de parada aprendida. [PonderNet](https://arxiv.org/abs/2107.05407) ya aprende
cuándo detener el cálculo. El contraste de MARS-TITAN distingue el coste y la utilidad
de renovar episodios maduros de los de seguir trabajando con los mismos episodios.
