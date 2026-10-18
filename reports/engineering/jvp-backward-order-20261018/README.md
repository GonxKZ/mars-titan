# Orden del backward interno de los JVP de C

La prueba `test_titans_pass_is_bit_identical_with_ranges[penalty-2-cuda:0]`
falló en el runtime `932d8ab02095c4bc9530ee5e3b97cbb8097a3171`.
Los controles off/off, on/on y off/on repiten nueve gradientes distintos
entre 407, con máximo absoluto 2,71e-20 en FP64. Las diferencias persisten
entre la segunda, tercera y cuarta pasada. La activación de NVTX no explica
por sí sola el fallo de igualdad.

Los [recibos del diagnóstico](diagnosis.json) corresponden a la fixture
técnica original, con un optimizador registrador que no actualiza parámetros.
No utilizan datos financieros reales ni miden un efecto predictivo.

## Aislamiento y alcance

Activar algoritmos deterministas de PyTorch o ejecutar solo el backward
exterior en el hilo llamador conserva las nueve diferencias. Ejecutar en ese
hilo únicamente los JVP obtiene igualdad exacta en off/off, off/on y cuatro
pasadas consecutivas. El mismo wrapper con multihilo activado reproduce el
fallo. En FP32, seis gradientes difieren con la política anterior y ninguno
con el contexto local. No se cambian `torch.equal`, semillas ni clipping.

En PyTorch 2.14.0+cu130, `autograd.functional.jvp` usa doble backward. Los
archivos instalados `torch/autograd/grad_mode.py`, `node.h` y `engine.h`
describen trabajadores de dispositivo y prioridad mediante números de
secuencia locales a cada hilo. El cambio observado es compatible con otro
orden de acumulación al construir el grafo de derivadas entre hilos.
No se ha identificado mediante una traza la primera suma divergente.
No se atribuye el fallo a `eigvalsh` ni a un kernel concreto.

Una sonda reducida con el operador de dimensión 32, rango 2, estado inicial
y pérdida cuadrática sin `eigvalsh` repite los bits con ambas políticas.
Sirve para medir ese operador, pero no reproduce el fallo. El recorrido
cronológico con replay y C sigue siendo el caso que lo reproduce.
Los controles con `local_control=None` y sin acumulación dieron igualdad
exacta en la fixture FP64. No acreditan toda una ejecución científica con
otra configuración.

## Corrección e identidades

`MACProjectionControl` limita `set_multithreading_enabled(False)` a cada JVP.
El contexto restaura el modo anterior también ante excepción. El radio y el
backward exterior conservan sus recorridos. Diagnóstico y penalización
declaran identidad 2 y `jvp_backward_execution='calling_thread_v1'`.
Sus checkpoints de identidad 1 se rechazan antes de copiar parámetros y
necesitan su runtime histórico para reanudarse.

`disabled` conserva exactamente la identidad 1 y carga el estado histórico
incluido en la fixture. Su siguiente salida, pesos rápidos y momentum
coinciden bit a bit. El emparejamiento B/C compara la configuración común
desactivada cuando alguno de los controles es B. Cada control conserva su
identidad y los controles activos siguen exigiendo la misma ejecución JVP.
La ruta con C ausente no cambia.

## Comprobaciones y coste

La selección CPU pasó 115 casos bajo cobertura y otro caso adicional de
rechazo de identidades activas distintas. Los nueve mutantes dirigidos se
detectaron en una copia aislada. [Cobertura y CRAP](cpu-quality.json)
registran 231/233 sentencias y 64/66 ramas del control local. El archivo
financiero completo tiene 361/461 sentencias y 110/196 ramas cubiertas en
esta selección. Las funciones modificadas `identity` y `_operator` tienen
CRAP 2, y el emparejamiento tiene 23,02. Estas medidas no demuestran ausencia
de errores. Los 166,44 segundos de la batería instrumentada no son una
medida de rendimiento del modelo.

La sonda del operador usa CUDA sincronizada, dos calentamientos y nueve
repeticiones por política, con orden alternado. Incluye construir el grafo,
pero excluye el gradiente de su pérdida cuadrática. Los recorridos completos
solo tienen dos pasadas por política, en procesos distintos y con trabajo
CPU simultáneo.

| Precisión | Operador con trabajadores | Operador en hilo llamador | Recorrido con trabajadores | Recorrido en hilo llamador |
| --- | ---: | ---: | ---: | ---: |
| FP64 | 39,409 ms | 38,140 ms | 7,274 s | 6,295 s |
| FP32 | 34,018 ms | 33,294 ms | 6,814 s | 7,407 s |

Son medianas descriptivas. Las dos pasadas completas no permiten afirmar
aceleración ni descartar una regresión. El proceso del operador alcanzó
1.539.592.192 bytes de RSS y 68.128.768 bytes asignados por CUDA, incluidas
las estructuras ya residentes. No son incrementos atribuibles al cambio.
Los recibos conservan hardware, versiones, flags, muestras y huellas.

La comprobación CUDA del código final y una medida alternada del recorrido
completo siguen pendientes. Las sondas sobre el runtime anterior no las
sustituyen.
