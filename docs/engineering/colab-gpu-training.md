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
| `scripts/colab_campaign.sh` | Lanza `comparison_campaign` con el entorno del repositorio. Exige `MARS_TITAN_SITE`, la etiqueta del equipo |
| `scripts/colab_site.py` | Ficha del equipo: máquina, GPU, controlador, CUDA, cuDNN y versiones. Compara dos fichas y dice en qué difieren |
| `scripts/colab_sync.py` | Copia coherente del estado de la campaña a Drive mientras los trabajos escriben, y su vuelta a una máquina nueva |
| `scripts/colab_parity.py` | Puerta de paridad entre equipos: mismas filas y etiquetas, y equivalencia del IC por sesión dentro de un margen |

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

## Orden previsto

1. Arranque y comprobaciones (`colab_bootstrap.sh`), ficha del equipo y prueba corta de
   concurrencia: uno, dos, cuatro y seis trabajos a la vez, mirando memoria del anfitrión, de
   la GPU y pasos por minuto.
2. Puerta de paridad con el commit de la campaña.
3. Criba de la base nueva (rama `v11-candidatos`) en el origen 2013 con tres semillas: sus
   opciones nacen apagadas y cada una se mide contra la base en el mismo equipo.
4. Ajuste final de lo que quede elegido, con todos los activos de EE. UU. y China y todas las
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
