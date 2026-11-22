# Entrenamiento en la GPU de Google Colab

La DGX entrena la campaña y hace la investigación. La GPU de Colab (RTX PRO 6000, con las
unidades de cómputo del plan de Google AI Ultra) es el segundo equipo: ajusta solo en GPU, con
varios modelos a la vez, lo que la investigación deje definido. Este documento dice qué hay
preparado, qué se comprobará allí antes de fiarse de un resultado y cómo queda escrito dónde se
ajustó cada modelo.

Nada de lo que sigue se ha ejecutado todavía en Colab. Lo que no se ha podido medir desde la
DGX está en [Sin verificar](#sin-verificar).

## Qué viaja y por dónde

| Qué | De dónde sale | Tamaño | Cómo llega |
| --- | --- | --- | --- |
| Datos de entrenamiento | El paquete de datos que ya está en Drive, el mismo del que se restauró la DGX el 20 de noviembre | 58,7 GB comprimidos, unos 90 GB restaurados | Se lee desde Drive dentro de Colab con su restaurador, que comprueba la huella SHA-256 de cada volumen |
| Código | Un bundle de Git con el commit autorizado | unos 45 MB | Es lo único que se sube desde la DGX, a la carpeta de Drive, y solo con el visto bueno del dueño |
| Entorno | `uv sync --locked --all-extras` | unos 3 GB de descarga | El bloqueo del proyecto ya es de Linux x86_64 con PyTorch 2.14.0 y CUDA 13.0, cuyo paquete incluye `sm_120` |
| Resultados | `export-results` y la copia del estado | megas por modelo | Vuelven a la DGX por Drive, sin pesos salvo que se pidan |

No se suben datos desde la DGX: la línea mide 3,4 MB/s de subida (22 de noviembre), así que las
copias empaquetadas de un solo origen (27 a 60 GiB) tardarían entre dos y cinco horas, y los
once orígenes, día y medio. Los grupos del paquete que entrenan son `A00-macroeconomia`,
`A01-preparados-v31`, `A02-etiquetas-v31`, `A03-muestras-codificadas-v31`, las vistas
`A04`, `A05` y `A06`, y los cuatro del kit de EE. UU. (`Muestras-US`, `Precios-US`,
`Etiquetas-US`, `Ventanas-US`). Los originales `B01` a `B07` no hacen falta.

El año 2024 no está en esos grupos: los precios preparados de los 4.202 activos de EE. UU. y de
los 810 de China terminan el 29 de diciembre de 2023 (auditoría del 22 de noviembre), y la
declaración de la campaña no admite orígenes posteriores a 2023.

## Herramientas

| Guion | Qué hace |
| --- | --- |
| `scripts/colab_bootstrap.sh` | Arranque por pasos repetibles: herramientas del sistema y uv, copia fija del código en su commit, entorno bloqueado, restauración de los datos con huellas, enlace episódico nativo y comprobaciones en CUDA |
| `scripts/colab_campaign.sh` | Lanza `comparison_campaign` con el entorno del repositorio. Exige `MARS_TITAN_SITE`, la etiqueta del equipo. Un commit anterior al kit (el de la campaña v10) se lanza con su `dgx_campaign.sh` y `MARS_TITAN_DGX_PYTHON` apuntando al entorno; sus intentos no llevan la etiqueta, pero sí la GPU, la plataforma y las versiones |
| `scripts/colab_site.py` | Ficha del equipo: máquina, GPU, controlador, CUDA, cuDNN y versiones. Compara dos fichas y dice en qué difieren |
| `scripts/colab_sync.py` | Copia coherente del estado de la campaña a Drive mientras los trabajos escriben, y su vuelta a una máquina nueva |
| `scripts/colab_parity.py` | Puerta de paridad entre equipos: mismas filas y etiquetas, y equivalencia del IC por sesión dentro de un margen |
| `scripts/colab_origin.sh` | Un origen paso a paso con las órdenes del relevo de la DGX: cohortes y comprobaciones, copia empaquetada de las entradas, permiso, cola y estado. Con `--ensayo` imprime cada orden sin ejecutarla |
| `scripts/campaign_search.py` | Búsqueda de hiperparámetros con Optuna en la que cada ensayo es un modelo más de una campaña. Sirve en cualquier equipo; en Colab necesita el paso `busqueda` del arranque |

El motor ya sabía trabajar en dos equipos
([guía de la campaña en dos equipos](comparison-campaign-two-machines.md)): `prepare`,
`doctor`, `authorize`, `run`, `status` y `export-results` son las mismas órdenes. En Colab
cada sesión dura como mucho un día, así que `colab_sync.py push` se ejecuta cada pocos minutos
hacia Drive y `colab_sync.py pull` deja el estado en la máquina siguiente. La copia publica
cada índice (registros de la cola, informes y puntos de recuperación) después de los archivos
que nombra, de modo que un corte a mitad de copia deja un estado anterior pero entero.

El servidor MCP oficial de Colab ([googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp),
1.0.1, Apache 2.0) está declarado en `.mcp.json` de `/home/gpic/TITANS-IA`, fijado al commit
`b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`. Abre un WebSocket en `localhost` con puerto y
testigo aleatorios y solo acepta la conexión del navegador con Colab abierto. No guarda
credenciales: la sesión de Google es la del navegador. Para usarlo hay que aprobarlo en la
sesión de Claude Code (`/mcp`) y tener la sesión de Google iniciada en el navegador.

## Procedencia

Cada intento de cada trabajo anota en su registro, además del commit y de la huella de la
declaración: la etiqueta del equipo (`site`, de `MARS_TITAN_SITE`), el modelo de GPU, su
arquitectura (`gpu_capability`), su memoria, el controlador, las versiones de CUDA y de cuDNN,
la plataforma y las versiones de Python, PyTorch, NumPy y PyArrow. Una etiqueta mal escrita
detiene el intento al empezar. Los estados de campaña de Colab van en una carpeta propia y sus
informes dirán de cada modelo dónde se ajustó. La ficha de la DGX está tomada
(`dgx-spark-gb10`: GB10, controlador 590.48.01, PyTorch 2.14.0+cu130, aarch64).

## Puerta de paridad

Un modelo ajustado en Colab no es comparable bit a bit con uno de la DGX: cambian la GPU, la
CPU (x86 frente a ARM) y el camino de cada operación. Antes de mezclar resultados de los dos
equipos en un mismo análisis:

1. Se ajusta el mismo modelo (`titans_mac`), en el mismo origen (2013), con las mismas semillas
   y **el mismo commit que la campaña de la DGX**, para que la única diferencia sea el equipo.
2. `colab_parity.py` comprueba que las filas y las etiquetas son idénticas y contrasta la
   diferencia del IC de rangos por sesión con error con autocorrelación y dos contrastes
   unilaterales dentro de ±0,002.
3. Las semillas comparten sesiones, así que se promedian por sesión: no cuentan como réplicas
   independientes. Con una sola semilla el contraste casi nunca concluye (el intervalo es del
   orden del margen); hacen falta las tres.

Si el veredicto no es «equivalente», los resultados de Colab se analizan aparte y cada origen
se ajusta entero en un solo equipo, con sus controles, para que ningún contraste pareado cruce
máquinas.

## Guion del primer día

Con Drive montado, `P` es la carpeta del paquete de datos, `B` el bundle de código y
`T=/content/mars-titan`. El kit (`K`) son los guiones de `scripts/` de la rama de trabajo, que
salen del mismo bundle. Cada orden se puede repetir: conserva lo ya hecho.

```bash
export MARS_TITAN_SITE=google-colab-rtx-pro-6000
# 0. El kit: la rama de trabajo del bundle, aparte de la copia fija que ejecuta la campaña.
[ -d $T/kit ] || git clone --quiet --branch dgx-trainning $B $T/kit
K=$T/kit/scripts
# 1. Máquina: herramientas, código de la campaña v10, entorno, datos de EE. UU. y comprobaciones.
bash $K/colab_bootstrap.sh sistema
bash $K/colab_bootstrap.sh codigo  --bundle $B --commit 3d10a30fcf8124a19b4aae74e9bb06e589e10986 --trabajo $T
bash $K/colab_bootstrap.sh entorno --trabajo $T
bash $K/colab_bootstrap.sh datos   --paquete $P --trabajo $T
bash $K/colab_bootstrap.sh nativo  --trabajo $T
bash $K/colab_bootstrap.sh comprobar --trabajo $T
# 2. Puerta de paridad: el núcleo en 2013 con las tres semillas, en el commit de la campaña.
origen=(--repo $T/repo --estado $T/estado/v10 --origen 2013
        --vistas /home/gonzalo/.local/state/mars-titan/campaign-a-v2-views-v31-20261013/views/US
        --declaracion configs/experiments/dgx-us-campaign-v10.json)
bash $K/colab_origin.sh cohortes "${origen[@]}"
bash $K/colab_origin.sh entradas "${origen[@]}" --receta configs/titans/dgx-us-core-v2.json
bash $K/colab_origin.sh permiso  "${origen[@]}" --nota "Puerta de paridad en Colab"
tmux new-session -d -s campana "bash $K/colab_origin.sh lanza ${origen[*]} --semillas '42 43 44' --niveles core --trabajadores 3"
# 3. Mientras corre: copia coherente a Drive cada pocos minutos y estado de la cola.
python3 $K/colab_sync.py push --state $T/estado/v10 --mirror $P/../estado-colab/v10
bash $K/colab_origin.sh estado "${origen[@]}"
```

De vuelta en la DGX, `colab_sync.py pull` deja el estado de Colab en una carpeta y
`colab_parity.py --a <estado de la DGX> --b <estado de Colab> --model titans_mac --origin 2013
--seed 42` (y las otras dos semillas) decide la puerta. El nivel `core` de la v10 lleva también
el lector `mars_titan_m3`, que necesita el enlace nativo: si el paso `nativo` no compila en
Colab, ese trabajo falla y el núcleo termina igual.

El número de trabajadores del primer lanzamiento es prudente. La prueba de concurrencia lo
sube: se relanza con cuatro y con seis mirando la memoria del anfitrión y de la GPU en
`equipo.json` y los pasos por minuto de la telemetría; un trabajo del núcleo usa en la DGX
hasta 20 GiB de memoria del anfitrión y 7,4 GiB de la GPU.

El paso `datos` restaura los once grupos que entrenan: unos 90 GB. Las tres familias de vistas
comparten las carpetas de datos preparados y de muestras codificadas, y cada una de esas
carpetas se reparte entre dos grupos (`A01` y `Precios-US`; `A03` y `Muestras-US`; las
etiquetas, entre `A02` y `Etiquetas-US`). Si el disco de Colab no da para todo, EE. UU. solo
necesita `--grupos "A00-macroeconomia A01-preparados-v31 A02-etiquetas-v31
A03-muestras-codificadas-v31 A04-vistas-US Etiquetas-US Muestras-US Precios-US Ventanas-US"`,
unos 80 GB: quedan fuera las vistas de China y las conjuntas. La orden `doctor` del paso
`cohortes` marca `FALTA` si a una cohorte le faltan datos.

La copia empaquetada de las entradas de 2013 ocupa 27 GiB, además de los datos restaurados.
Para la criba de la base nueva se repite el paso `codigo` con el commit de `v11-candidatos`
en otra carpeta de trabajo (`--trabajo $T-v11`), que comparte los datos restaurados, y se usa
su declaración `configs/experiments/us-campaign-v11-screen1.json` con otro estado.

## Búsqueda de hiperparámetros

La criba mide cada campo de uno en uno. `scripts/campaign_search.py` busca varios a la vez:
Optuna propone los valores y la campaña entrena, así que cada ensayo conserva la identidad, la
telemetría y la procedencia de cualquier otro trabajo.

- La declaración (`campaign_search_v1`) nombra la campaña de partida, el modelo base, el origen
  y la semilla que se miran, y el espacio: campos de la receta con un intervalo (`float`, lineal
  o logarítmico) o con una lista ordenada de valores (`ordinal`). El objetivo no se declara: es
  siempre el IC de rangos de validación de ese origen. La primera declaración es
  `configs/experiments/search/us-v11-search1.json`, en la rama `v11-candidatos`.
- `ask` pide una tanda. Los primeros ensayos (`startup`) son un hipercubo latino del espacio,
  fijado por la semilla; después propone el estimador de Parzen en árbol multivariante, que
  cuenta con los ensayos que siguen en marcha. Escribe una receta por ensayo en
  `configs/titans/search/<búsqueda>/`, y en `configs/experiments/search/` la campaña de la tanda
  (el modelo base y todos los ensayos vivos hasta ella) y sus controles. Tras el commit, la
  tanda se ejecuta como cualquier criba: `plan`, `prepare`, `inherit` desde el estado de la
  tanda anterior y `run --tier search_g<N> --origin <año> --seed <semilla>`.
- `tell` lee de ese estado los ensayos terminados. A Optuna le da la validación; guarda además
  la calibración y el contraste pareado por sesión de cada ensayo con el modelo base. El tramo
  final del origen no se abre en ningún momento. El modelo base entra en el estudio como punto
  conocido cuando está medido en el mismo estado, es decir, en el mismo equipo que los ensayos.
- `report` ordena por validación y confirma en calibración a los mejores: un candidato vale si
  también gana al modelo base en calibración. Informa de la concordancia de rangos entre los
  dos tramos y de cuánto ganaría por azar el mejor de `n` ensayos sin efecto, para leer la
  ventaja del ganador contra ese listón. Un candidato confirmado se repite con las otras dos
  semillas antes de adoptarlo.
- `fail` da por perdido un ensayo que no va a terminar, para que deje de contar como pendiente
  y no entre en la tanda siguiente.

La carpeta de la búsqueda (`--study`) guarda el diario de Optuna, la huella de la declaración
y el registro de ensayos. Cambiar el espacio, la semilla o el modelo base exige otra búsqueda;
cambiar cuántos se confirman o qué modelos se dan por conocidos, no.

## Orden previsto

1. Arranque y comprobaciones (`colab_bootstrap.sh`), ficha del equipo y prueba corta de
   concurrencia: uno, dos, cuatro y seis trabajos a la vez, mirando memoria del anfitrión, de
   la GPU y pasos por minuto.
2. Puerta de paridad con el commit de la campaña.
3. Criba de la base nueva (rama `v11-candidatos`) en el origen 2013 con tres semillas: sus
   opciones nacen apagadas y cada una se mide contra la base en el mismo equipo.
4. Si quedan unidades de cómputo después de reservar el ajuste final, la búsqueda conjunta
   `us-v11-search1`: ocho ensayos de arranque y tandas de seis.
5. Ajuste final de lo que quede elegido, con todos los activos de EE. UU. y China y todas las
   modalidades, en los orígenes limpios.

## Sin verificar

- La tarifa en unidades de cómputo por hora del entorno con la RTX PRO 6000. Se lee en el
  panel de Colab al conectar; con ella, las 1.000 unidades son `1000 / tarifa` horas de GPU.
- La memoria, los núcleos y el disco de la máquina de Colab, de los que depende cuántos
  trabajos caben a la vez (en la DGX un ajuste usa entre 4 y 17 GiB de memoria del anfitrión).
- Si Colab permite CUDA MPS. Sin él, varios procesos comparten la GPU por turnos.
- La escritura de memoria fusionada (Triton) en `sm_120` sobre x86_64: se desarrolló en la
  GB10. El paso `comprobar` ejecuta sus pruebas antes de ajustar nada.
- La compilación del enlace episódico en la máquina de Colab. Los ajustes del núcleo y
  `mars_titan_m0` no lo necesitan; los lectores m1, m2 y m3, sí.

## Decisiones del dueño antes de empezar

1. Confirmar la carpeta de Drive que contiene el paquete de datos del 20 de noviembre.
2. Autorizar la subida del bundle de código a esa carpeta.
3. Elegir en Colab el entorno con la RTX PRO 6000 y aprobar el servidor `colab-mcp`.
4. Fijar el techo de unidades de cómputo y la reserva.
5. Decir qué se ajusta primero después de la paridad: la criba de la base nueva o el ajuste
   final de la base actual.
