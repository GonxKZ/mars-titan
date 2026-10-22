# Intervenciones sobre lectores y puentes regionales

Esta ampliación de #710 permite silenciar lectores y cortar conexiones de SharedEvidenceBridge sin modificar el registro episódico canónico. Conserva un único EpisodeSnapshot, sus IDs, valores, etiquetas maduras y procedencia. La implementación y los contrastes técnicos tienen un [recibo propio](../../reports/engineering/regional-bridge-interventions-20261022/README.md). La evidencia anterior del puente lineal y del bucle no lineal se conserva por separado.

## Configuración y operación

`EvidenceBridgeConfig.intervention` admite `None` o una configuración fija:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BridgeInterventionConfig:
    active_regions: tuple[bool, ...]
    removed_edges: tuple[tuple[int, int], ...] = ()
```

`active_regions` contiene un booleano por región, en el orden de `config.regions`. No es una lista de nombres ni cambia el número de regiones. Su longitud debe coincidir con R, entre una y ocho. Los enteros 0 y 1 no sustituyen a los booleanos. Se puede declarar que todas las regiones estén inactivas.

Cada corte es una pareja de índices enteros `(receptor, emisor)`, empezando en cero. Las parejas deben ser únicas, estar ordenadas y pertenecer a la topología base. No se añaden conexiones ni se modifica `edges` para construir una condición de prueba. La forma JSON declara ambos campos y convierte sus secuencias a las tuplas tipadas. Por ejemplo, para tres regiones y un anillo:

```json
{
  "active_regions": [true, false, true],
  "removed_edges": [[0, 0], [2, 0]]
}
```

El corte `[2,0]` impide enviar desde la región 0 a la 2, no al contrario. También se puede retirar una autoarista, como `[0,0]`. Ese corte afecta al mensaje del grafo y no elimina el término residual de la región.

La máscara efectiva por fila es `m[b,r] = available[b,r] AND active_regions[r]`. Se aplica a las atenciones regionales, las vistas iniciales, las actualizaciones y los emisores y receptores del grafo. Para la adyacencia base E y los cortes D:

$$
Q_{brs}=E_{rs}\,1[(r,s)\notin D]\,m_{br}m_{bs},
\qquad G_{brs}=\frac{Q_{brs}}{\max(1,\sum_sQ_{brs})}.
$$

Los cortes se aplican antes de normalizar. Una fila sin mensajes queda a cero y no recibe una conexión de sustitución. En el recorrido no lineal se conserva `R_next=(1-g)R+g*tanh(message W_r)`, con la máscara de presencia aplicada después. Si una región permanece activa y pierde todas sus aristas, su mensaje es cero, pero conserva `(1-g)R`. Con g=1 ese término también es cero. Silenciar el lector completo y desconectarlo son intervenciones distintas.

Las validaciones de formas, identidades y finitud siguen precediendo al acceso o a la operación que protegen. Una máscara no debe ocultar NaN, Inf o un desbordamiento antes de tanh. Tampoco altera la disponibilidad temporal de la evidencia.

## Identidad, presencia y recuperación

`intervention=None` y `intervention:null` representan la opción desactivada. Omiten el nuevo campo en la identidad y no añaden el buffer `active_regions`. Una intervención explícita con todas las regiones activas y sin cortes tiene identidad propia aunque su efecto numérico sea nulo.

La identidad activa utiliza `schema_version=3`, `kind="shared_evidence_bridge_intervention_v1"`, el `base_kind` anterior y `asdict(intervention)`. Así distingue la base lineal de la no lineal y conserva J y los demás parámetros declarados. El lector declara `regional_availability="fixed_declared_active_views"`. El consumidor utiliza esquema 4 y la receta `frozen_intervened_regional_mechanism_cohort_v1`. El catálogo debe corresponder al lector y a esa intervención, no a una condición anterior con la misma forma de tensores.

La intervención no añade parámetros entrenables ni cambia la secuencia de inicialización. El emparejamiento debe comprobar las huellas de los parámetros y los estados RNG, no deducirse únicamente de compartir semilla. La restauración debe validar campos, identidad, precisión, formas y contenido declarado de los buffers antes de copiar parámetros. La referencia para `edges` y `active_regions` debe derivarse de la configuración, sin aceptar como autoridad un buffer mutable previamente alterado.

La identidad estructural desactivada y el golden legado deben conservarse, pero las fuentes y los sellos de runtime cambian cuando cambia el código. Eso no autoriza a recuperar entre runtimes distintos por tener configuraciones o tensores con igual forma. La recuperación completa necesita comparar sus sellos y utilizar el runtime compatible. No se cambian retroactivamente las identidades de ejecuciones históricas.

En el lector regional, `presence` es verdadera si hay candidatos y alguna región efectiva activa. Banco ausente, snapshot vacío y banco poblado con todos los lectores silenciados son casos distintos. En este último se mantienen los IDs comunes y la huella del snapshot, con atenciones, vistas, grafo y lectura a cero. Una lectura cero no implica una predicción final cero: el refinador y la cabeza conservan su recorrido propio. Tampoco una salida cero por cancelación equivale a presencia falsa.

Las trazas y los recibos deben vincular configuración, máscara efectiva, condición de lectura, corte y snapshot con la emisión real. Las regiones activas con candidatos tienen suma de atención uno y las silenciadas suma cero. El grafo efectivo capturado procede de `BridgeRead.bridge_weights`. Si se reconstruye desde la configuración, se identifica como reconstruido. No se atribuyen consultas capturadas a formatos que no las contienen.

La intervención no cambia retención M0–M3, madurez o procedencia del error financiero. La sesión mantiene una sola confirmación y recibe el resultado de la emisión realmente realizada bajo esa condición. Las alternativas descartadas no publican otra memoria ni duplican feedback.

## Contrastes analíticos fijados antes de ejecutar

El primer panel usa dos episodios artificiales con IDs 1 y 2, claves e₁ y e₂ en R⁶⁴ y valores iniciales `(1,0)` y `(0,1)`. Las etiquetas son cero y la proyección de valores no las utiliza. Se fijan disponibilidad 1, decisión 2, madurez 3 y corte 4. No son observaciones financieras. Se mantiene K=1 y precisión FP64.

Con H=2 y temperatura `1/ln(3)`, las consultas regionales e₁, e₂ y `(e₁+e₂)/sqrt(2)` producen atenciones `(3/4,1/4)`, `(1/4,3/4)` y `(1/2,1/2)`. Las matrices, aplicadas a vectores fila, son:

$$
W_1=\begin{pmatrix}4/3&-1/3\\0&1\end{pmatrix},\quad
W_2=\begin{pmatrix}1&0\\-1/3&4/3\end{pmatrix},\quad
W_3=\begin{pmatrix}2&0\\0&-2\end{pmatrix}.
$$

Para a=tanh(1), las vistas iniciales son `(a,0)`, `(0,a)` y `(a,-a)`. Proceden de los mismos dos episodios con atenciones y transformaciones diferentes. No se usa un único candidato que haga trivial la atención.

Fijar J=1, g=1/2 y objetivo de reconstrucción `t=(1,1)`, calculado como suma del contenido original. Sean `b=tanh(2a/3)` y `c=tanh(a/3)`. La salida receptora es `O=[D 0 0]`, con `D=diag(2/(a+b),2/c)`. Sus coeficientes se declaran por la derivación, sin ajustarlos a una ejecución observada. La pérdida técnica es `E=||y-t||²`.

| Condición predefinida | Referencia analítica contrastada con fixtures |
| --- | --- |
| Anillo con las tres regiones | La región 1 recibe las vistas 1 y 2 y produce `(1,1)`, con E=0 en aritmética real. |
| Retirar la región 2 o cortar su mensaje hacia la 1 | La salida receptora pierde esa contribución y coincide con el control `self`. |
| Retirar la región 3 en el anillo | La salida receptora conserva t para este único paso. |
| Grafo completo | La tercera vista antagonista empeora la reconstrucción con la misma salida. Más comunicación no garantiza menor error. |
| Tres lectores con consulta y matriz iguales a las de la región 1 | La redundancia simple no introduce la vista complementaria. Deben contarse los parámetros almacenados y su menor diversidad funcional. |
| Control fuerte sin comunicación, `O_direct=(1/a)[I₂ I₂ 0]` | Reconstruye t exactamente. Impide interpretar el éxito del anillo con salida receptora fija como necesidad universal o superioridad de los puentes. |

La retirada aleatoria debe usar una semilla local declarada y guardar la máscara o los cortes resueltos antes de observar resultados. Se retira el mismo número de lectores o conexiones que en su control dirigido. La recuperación no vuelve a sortear y los controles reutilizan la misma intervención. Los cortes dirigidos se contrastan también sobre un grafo completo, para distinguir receptor y emisor sin confundir una arista inexistente con un corte válido.

Un segundo fixture mantiene los episodios y utiliza dos vistas opuestas `(a,0)` y `(-a,0)`, con objetivo `(1,0)`. La media completa con g=1 cancela ambas vistas y produce cero, mientras el snapshot conserva sus valores e IDs. El control lineal sin comunicación recupera el objetivo. Retirar la segunda vista y renormalizar puede restaurarlo en ese control lineal, sin atribuir la misma igualdad al recorrido no lineal que vuelve a aplicar W y tanh. Para R=2, `ring` y `complete` tienen la misma adyacencia y no se cuentan como tratamientos diferentes.

Borrar un episodio exige otro snapshot y otra huella. Dos fuentes que solo difieren en el episodio borrado pueden dejar exactamente el mismo input restante y tener objetivos de reconstrucción distintos. Un lector determinista sin información adicional no puede recuperar ambos. Este límite no se infiere del fallo de una única cabeza elegida ni desaparece por conservar otros lectores.

## Coste y comprobaciones realizadas

Las condiciones emparejan snapshot, candidatos, semilla y parámetros. Cambiar la máscara o cortar aristas no reduce R, J o K ni el trabajo denso declarado. Con banco poblado se conservan las búsquedas y la proyección común, incluso si todos los lectores están inactivos. Las vistas iniciales se cuentan con `R × value_projections` y las actualizaciones nuevas con `R × J × value_projections`. No son FLOPs. La reserva conjunta mantiene 38 lecturas conservadoras y hasta 22 con prefijos compartidos. No se renueva por ruta, bloque o intervención.

Compartir evidencia no elimina temporales ni garantiza ahorro de RAM, allocator o VRAM física. Deben añadirse los bytes de máscaras, buffers y trazas al presupuesto y medirse el recorrido completo. El primer panel fija K=1 para conservar los candidatos comunes. Las rutas que renuevan consultas con K>1 pueden elegir otros IDs después de intervenir y necesitan un contraste separado.

El informe registra el alcance de estas comprobaciones, sus herramientas y las limitaciones de cobertura y coste:

- [x] Paridad del modo ausente y `null` con el golden anterior, identidad propia del control explícito sin efecto y parámetros/RNG emparejados.
- [x] Constructor, JSON y factoría equivalentes, validación de tipos, longitud, cortes únicos y ordenados, índices, topología base y autoaristas.
- [x] Máscara efectiva en atención, vistas y mensajes, cortes antes de normalizar y residual conservado al desconectar una región activa.
- [x] Referencias analíticas, control fuerte sin puentes, redundancia simple, cancelación, lesión dirigida y lesión aleatoria fijada previamente.
- [x] Banco ausente, vacío y poblado con todos los lectores inactivos, con presencia, IDs, huellas y lectura coherentes.
- [x] Gradientes y finitud CPU/CUDA pertinentes, orden y partición de filas, aislamiento de estados y snapshots sin cambios.
- [x] Rechazo de checkpoint, buffer o catálogo incompatibles antes de copiar, incluido un buffer alterado antes de congelar el consumidor.
- [x] Trazas coherentes, límites densos, una única confirmación y recuperación de la siguiente emisión con sus sellos de runtime y feedback real.
- [x] Cobertura de sentencias y ramas, complejidad y CRAP con herramientas y convención, mutación dirigida con control inicial y restaurado y medidas de tiempo, caudal, memoria y bytes.

La [revisión de fuentes y extensiones](../research/regional-recurrence-extensions.md) recoge el alcance ya verificado de [Roy y colaboradores (2022)](https://www.nature.com/articles/s41467-022-29384-4.pdf) y [Schmitt y colaboradores (2017)](https://hearingbrain.org/docs/schmitt_halassa_nature_2017.pdf). Motivan preguntas sobre cooperación y conectividad, no la ecuación o los resultados de estos fixtures. Las reconstrucciones anteriores son derivaciones propias y contrastes técnicos ejecutados. No acreditan especialización financiera, recuperación perfecta o ventaja predictiva. Tampoco implementan los grupos jerárquicos de #709 o el proponente inspirado en SEAL de #711. La ampliación permanece desactivada por defecto y la campaña completa y el test final siguen cerrados.
