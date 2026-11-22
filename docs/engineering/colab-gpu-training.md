# Entrenamiento en la GPU de Google Colab

La DGX entrena la campaña y hace la investigación. La GPU de Colab (RTX PRO 6000, con las
unidades de cómputo del plan de Google AI Ultra) queda preparada para la última fase: entrenar
solo en GPU, con varios modelos a la vez, los modelos que la investigación deje definidos.
Este documento dice qué está instalado, qué falta y cómo se anotará la procedencia.

## Qué está instalado

El servidor MCP oficial de Colab ([googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp),
versión 1.0.1, licencia Apache 2.0) está declarado en `.mcp.json` del proyecto
`/home/gpic/TITANS-IA`, fijado al commit `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`. Se revisó su
código antes de instalarlo: abre un WebSocket en `localhost` con un puerto y un testigo
aleatorios y solo acepta una conexión, la del navegador con Colab abierto
(`colab.research.google.com` o `colab.google.com`). No guarda credenciales: la sesión de Google
es la del navegador. Sus registros quedan en `~/.local/state/mars-titan/colab-mcp/logs`.

Comprobado: el servidor arranca y responde a `initialize` y a `tools/list` con su única
herramienta inicial, `open_colab_browser_connection`. Las herramientas del cuaderno (crear
celdas, ejecutar código) aparecen cuando el navegador se conecta.

## Qué falta para vincularlo

1. Aprobar el servidor `colab-mcp` al abrir la sesión de Claude Code en el proyecto (o con
   `/mcp`): una sesión ya abierta no carga servidores nuevos.
2. Tener iniciada la sesión de Google en el navegador de esta máquina. La primera llamada a
   `open_colab_browser_connection` abre un cuaderno vacío de Colab que se conecta al servidor.
   Si se trabaja desde otro equipo, el navegador de ese equipo necesita el puerto del servidor
   redirigido a esta máquina.
3. Elegir en Colab el entorno con la RTX PRO 6000.

## Qué se ejecutará allí y qué no

- **Sí:** los ajustes finales de los modelos y variantes que la campaña de la DGX deje
  elegidos, con todos los activos (EE. UU. y China) y todas las modalidades, y la búsqueda de
  hiperparámetros con poda, que necesita muchos ajustes cortos a la vez.
- **No:** nada que decida entre modelos con el tramo de evaluación, ni el año 2024, que sigue
  sellado. Las reglas causales y de precisión (FP32 IEEE, algoritmos deterministas) son las
  mismas que en la DGX.
- **Antes de subir datos** se confirma con el usuario qué se copia a su cuenta de Google: las
  copias empaquetadas de cada origen ocupan entre 27 y unos 60 GiB.

## Procedencia

Un modelo entrenado en Colab no es comparable bit a bit con uno de la DGX: cambian la GPU
(arquitectura y controlador), la CPU (x86 frente a ARM) y las bibliotecas. Por eso:

- Cada ejecución anotará en su `run.json` el equipo (`google-colab` o `dgx-spark-gb10`), el
  modelo de GPU, el controlador, las versiones de CUDA y de PyTorch, el commit del código, la
  huella de la declaración y de los datos, y las unidades de cómputo consumidas.
- Los estados de campaña de Colab irán en una carpeta propia y sus informes llevarán el equipo
  en el nombre. Ningún contraste pareado mezclará modelos de equipos distintos sin decirlo.
- Para pasar de un equipo a otro se exige el nivel 1 de la puerta de precisión: paridad
  matemática en FP64 de las piezas y equivalencia estadística pareada con varias semillas
  dentro de la diferencia mínima importante, medida con un mismo modelo entrenado en los dos.
- El documento final dirá de cada modelo dónde se entrenó.

Pendiente de implementar: el bloque de procedencia del equipo en `run.json` y el empaquetado
de una campaña (copia fija del código, declaración, cohortes y copias de entrada) para
llevarla a Colab.
