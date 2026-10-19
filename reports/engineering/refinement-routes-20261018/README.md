# Verificación y coste de las rutas de refinamiento

La implementación de [#56](https://github.com/GonxKZ/mars-titan/issues/56) tiene comprobaciones
de componentes y recorrido cronológico en CPU, seguidas de pruebas CUDA de las nuevas rutas
y controles. Los recibos iniciales conservan el estado de su revisión. La ampliación CUDA
se describe debajo. No contiene resultados predictivos.

El bloque de regresión supera 237 pruebas en 225,43 segundos, con ocho casos CUDA separados
para su ejecución coordinada. La mutación dirigida detecta 14 defectos y el bloque repetido
tras restaurar las fuentes supera 83 pruebas. Esos grupos se solapan y no deben sumarse.
Las fuentes principales corresponden a `89a0d07d86f14f994058286faf7f20e8e95d8044`.
El benchmark conserva sus propias huellas de fuente. La ampliación posterior de controles
por umbral y hash tiene otro recibo y no se incluye en estos recuentos.

Las mutaciones cubren renovación de episodios, número de refinamientos, presupuesto,
reutilización de proyecciones y gradientes, coste de la puerta, soporte, signo de las
ganancias, fronteras temporales, ponderación por sesión y valores no finitos. Se aplicaron
en una copia privada y se restauraron antes de repetir las pruebas. El resultado 14/14
no es una campaña exhaustiva de mutación del repositorio.

Coverage.py 7.16.1 registra 1.270 de 1.376 sentencias y 367 de 470 ramas en seis módulos.
Radon 6.0.1 y el resumidor del proyecto miden 105 funciones. El máximo CRAP es 45,0334 en
`MarsTitanInference.__init__`. CRAP utiliza cobertura de sentencias y la fórmula
`C² × (1 − cobertura)³ + C`. Las ramas se informan por separado. La cobertura restante
incluye caminos CUDA y errores no recorridos y no acredita ausencia de defectos.

## Medida del componente en CPU

La medición usa nueve combinaciones de lote 1, 64 o 256 y banco vacío, de 128 o de 1.024
episodios. Cada combinación tiene tres calentamientos y treinta repeticiones con orden
rotatorio. Se utiliza FP32 sin TF32, dos hilos y datos técnicos reproducibles. Había una
batería general y otras comprobaciones CPU en paralelo. Los tiempos no son una medida de
una máquina aislada ni del rendimiento de la RTX.

Con 128 episodios, las medianas en milisegundos son:

| Filas | K4 `first_read` anterior | K4 con caché | K1 y cabeza | Puerta que devuelve K1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,4747 | 1,1137 | 0,5943 | 1,0919 |
| 64 | 4,1209 | 2,6886 | 1,6358 | 2,2681 |
| 256 | 15,8725 | 10,8901 | 7,1135 | 8,3497 |

La caché y el K4 anterior producen los mismos bits en las comprobaciones del benchmark.
Las dos primeras columnas miden el lector y devuelven su estado. Las otras dos incluyen
la cabeza. La puerta también calcula sus rasgos y decide, por lo que añade coste incluso
cuando termina en K1. Una reducción de error o de otros refinamientos debe justificar ese
coste antes de incorporarla a un modelo.

El proceso completo de esta parrilla dura 15,162 segundos y alcanza 623.673.344 bytes de
RAM. Ese pico incluye intérprete, bibliotecas, modelos y referencias de comparación. No
se atribuye al tamaño de una caché. No se han medido energía ni coste monetario.

Los tiempos no incluyen preparar Titans-MAC, cargar una sesión financiera ni entrenar.
Los percentiles p95/p99 y las repeticiones completas están en el JSON y son descriptivos
de esa muestra. La comprobación del recorrido completo sigue siendo necesaria antes de
afirmar una aceleración del predictor.

Recibos: [verificación](verification.json), [cobertura y CRAP](quality.json),
[mutaciones](mutations.json) y [coste del componente](cpu-component-cost.json).

## Comprobaciones y coste en CUDA

Cinco pruebas CUDA comprueban paridad de rutas y gradientes en FP32 y FP64, rechazo atómico
de una carga inválida y controles de umbral y hash. Los umbrales conservan su comparación
en FP64, los códigos coinciden con la referencia y los RNG permanecen intactos. No hay
pasos de optimizador en esos cinco casos. La regresión integrada del commit `d8d98f9a1`
supera veinte pruebas CUDA en 31,95 segundos de pytest. Incluye estos cinco casos y las
pruebas heredadas del lector, la inferencia y la disposición de atención. No se suman como
veinticinco pruebas distintas. Los recibos fijan las huellas de todos los archivos probados.

La medida utiliza la RTX 4070 Laptop, PyTorch 2.14.0+cu130 y FP32 sin TF32. Hay cinco
calentamientos y treinta repeticiones con orden rotatorio por combinación. Otras pruebas
CPU seguían en marcha. Se midieron tanto las comprobaciones inmediatas como las agrupadas
que utiliza el recorrido cronológico. Agruparlas conserva su comprobación al terminar
la llamada, sin desactivar los controles de finitud.

Con comprobaciones agrupadas y 128 episodios, las medianas en milisegundos son:

| Filas | K4 `first_read` | K4 con caché | K1 y cabeza | Puerta que devuelve K1 | Rutas mezcladas explícitas |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,6732 | 2,9888 | 1,5515 | 2,7590 | 2,1496 |
| 64 | 2,8870 | 2,1609 | 1,1666 | 1,9863 | 8,8789 |
| 256 | 2,8194 | 2,1019 | 1,1557 | 2,0438 | 8,8410 |

K4 con caché conserva los bits de `first_read` en esta comparación. En el lote de 64 filas,
la mediana baja un 25,15 % y el pico de tensores CUDA pasa de 35.697.664 a 34.823.680 bytes.
Ese contador del framework no incluye toda la memoria del contexto CUDA. La puerta de
este fixture tiene buffers FP32. El estudio debe medir la política realmente ajustada,
incluidas su precisión y sus frecuencias de ruta.

Las rutas mezcladas muestran el coste de dividir el lote y ejecutar sus continuaciones.
La puerta también añade coste cuando elige K1. Estas medidas no justifican incorporarla
sin comprobar su error predictivo y el recorrido completo. Los tiempos de extracción,
lectura de datos y preparación MAC quedan fuera de este benchmark.

El benchmark supera quince pruebas CPU y cuatro mutaciones dirigidas. Coverage.py 7.16.1
y Radon 6.0.1 registran 147/150 sentencias, 37/40 ramas y CRAP máximo 25 en once funciones.
La cobertura incluye una ejecución CUDA pequeña instrumentada. Sus tiempos se excluyen
de las cifras anteriores. Los recuentos no se suman a los de las pruebas iniciales.

Recibos nuevos: [verificación CUDA](cuda-verification.json),
[coste con controles inmediatos](cuda-cost-immediate.json),
[coste con controles agrupados](cuda-cost-deferred.json),
[calidad del benchmark](benchmark-quality.json) y [mutaciones](benchmark-mutations.json).
