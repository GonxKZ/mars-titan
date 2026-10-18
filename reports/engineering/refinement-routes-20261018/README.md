# Verificación y coste de las rutas de refinamiento

La implementación de [#56](https://github.com/GonxKZ/mars-titan/issues/56) tiene comprobaciones
de componentes y recorrido cronológico en CPU. Las verificaciones CUDA y la evaluación
financiera de la puerta siguen pendientes en este recibo. No contiene resultados predictivos.

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
de esa muestra. La comprobación del recorrido completo y la medida CUDA son necesarias
antes de afirmar una aceleración del predictor.

Recibos: [verificación](verification.json), [cobertura y CRAP](quality.json),
[mutaciones](mutations.json) y [coste del componente](cpu-component-cost.json).
