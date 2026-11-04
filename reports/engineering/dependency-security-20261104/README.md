# Dependencias corregidas y comprobaciones de compatibilidad

La revisión del 4 de noviembre actualiza seis dependencias que tenían avisos en el lock.
La consulta de `uv audit --frozen` pasa de 80 avisos a cero sobre los mismos 192 paquetes.
El [inventario](advisories.json) conserva los identificadores, las versiones y los enlaces
de los avisos.

| Paquete | Versión anterior | Versión corregida |
| --- | --- | --- |
| certifi | 2026.7.22 | 2026.10.18 |
| filelock | 4.0.0 | 4.2.1 |
| nbconvert | 7.17.1 | 7.17.2 |
| pillow | 12.3.0 | 13.0.0 |
| setuptools | 84.0.0 | 84.1.0 |
| tornado | 6.5.10 | 6.6 |

Solo cambian esas seis versiones de dependencias. El paquete editable del proyecto se
reinstala con la misma versión. Torch, NumPy y PyArrow mantienen sus versiones. La
sincronización se hace con uv y `--inexact`, y `uv pip check` confirma la compatibilidad de
los paquetes instalados.

El corte general de resolución sigue en el 26 de septiembre. Se añaden mínimos corregidos
y una excepción fechada al 4 de noviembre para cada uno de los seis paquetes. No se abre
la resolución del resto de bibliotecas. El lock pasa su comprobación de consistencia.
La [configuración de uv](https://docs.astral.sh/uv/reference/settings/#exclude-newer-package)
describe el alcance de estas excepciones. Las
[notas de Pillow 13](https://github.com/python-pillow/Pillow/blob/main/docs/releasenotes/13.0.0.rst)
recogen sus correcciones de seguridad.

## Evidencia ejecutada

Las regresiones de imágenes, ventanas, presupuestos de codificación y exportación pasan
341 casos. Las comprobaciones numéricas focales y de
referencia pasan 224 casos. No hay fallos ni omisiones en
estas selecciones. Los tres controles OHLC sintéticos conservan exactamente PNG, píxeles
y vectores entre Pillow 12.3 y 13. No se reescriben ediciones anteriores.

También se comprueban exclusión y nueva adquisición de un bloqueo, carga del almacén de
certificados, metadatos de setuptools, una corrutina de Tornado y exportación HTML de un
cuaderno. Esta comprobación de las API no ejecuta el cuaderno ni carga pesos preentrenados.

Las seis [mutaciones de configuración](mutations.json) retiran por separado un mínimo
corregido. El resolver acepta entonces la versión vulnerable solicitada, lo que incumple
la política. Los controles original y restaurado la rechazan. Los códigos de salida del
resolver se distinguen del resultado de la comprobación de esa política.

Ruff, formato y `scripts/check_repository.py` pasan. No cambian funciones del modelo, por
lo que no se atribuye a esta actualización una nueva medida de CRAP o cobertura del código.
El [recibo](verification.json) conserva los resultados, las versiones y las huellas.

## Alcance

La protección de aprendizaje permanece vigente. No hay pasos de optimizador, estudios
científicos nuevos ni acceso al test final. La actualización no demuestra una mejora
predictiva y los runtimes históricos permanecen separados. Los cero avisos se refieren
a la consulta realizada. La paridad de tres gráficos no certifica todos los formatos de
imagen ni todas las transformaciones de las bibliotecas.
