# Enrutamiento de B6 por régimen: coste y mutación dirigida

Recibos del 10 de octubre de 2026 para #20. No se ha ajustado ni evaluado nada, no se han
usado datos de mercado y no se ha ejecutado ningún paso de optimizador. El diseño, las pruebas
y la comparación declarada están en la [guía de MARS-TITAN con ampliaciones](../../../docs/engineering/mars-titan-extensions.md#enrutamiento-de-b6-por-régimen).

## Coste por evento

[`cost.json`](cost.json) mide en CPU, con dos hilos, tres calentamientos y veinte
repeticiones, la ruta, las claves, la lectura de A y la escritura proximal de una cohorte de
64, 128, 1.024 y 4.096 filas con las claves `constant`, `regime`, `codec` y
`codec_by_regime`. Las ventanas, las entradas del codec y los valores son sintéticos con
semilla fija y solo fijan las formas reales. La comparación con el núcleo usa el caudal de
2.607 filas por segundo medido en la fold-012 con una vista v3 que ya no existe, así que las
fracciones solo sirven para ordenar el coste.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 uv run --no-sync python \
  benchmarks/regime_routing_cost.py --core-rows-per-second 2607 --output <recibo nuevo>
```

| Cohorte | Evento con `regime` | Evento con `codec_by_regime` | Añadido frente a `constant` / `codec` | Fracción del evento del núcleo |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0,36 ms | 2,68 ms | 0,19 ms / 2,30 ms | 0,8 % / 9,4 % |
| 128 | 0,42 ms | 3,19 ms | 0,22 ms / 2,82 ms | 0,4 % / 5,7 % |
| 1.024 | 2,48 ms | 8,81 ms | 1,96 ms / 7,39 ms | 0,5 % / 1,9 % |
| 4.096 | 11,41 ms | 33,66 ms | 9,02 ms / 28,33 ms | 0,6 % / 1,8 % |

La ruta cuesta lo mismo con las dos claves y crece con la cohorte (8,7 ms con 4.096 filas,
sobre todo la mediana de NumPy). Con `codec_by_regime` domina la escritura proximal con 320
coordenadas, una factorización de Cholesky de LAPACK de 320 × 320 más el producto KᵀWK.
Resolver solo los compartimentos con etiquetas la acercaría a la del codec, pero ahorraría
como mucho unas diez horas de GPU entre los dos modelos en A y no se ha implementado. El recibo
es la medida repetida después de corregir la tendencia (la mediana del rendimiento de la
ventana en lugar de la suma de las medianas diarias), con una carga media del equipo de 20
por otros procesos. Las dos medidas anteriores del mismo día, con cargas parecidas, dieron
cifras hasta un 50 % distintas en ambos sentidos, así que la dispersión entre ejecuciones es
grande y las cifras solo ordenan el coste.
No se midió en GPU porque la ruta y A trabajan en CPU y FP64, ni la energía, por falta de
instrumento.

## Mutación dirigida

[`mutations.json`](mutations.json) recoge 28 mutaciones aplicadas de una en una sobre una
copia aislada de `src`, `tests` y `configs`, con las pruebas importando la copia. Cada entrada
guarda el fragmento original y el mutado. Las 28 hicieron fallar
`tests/memory/test_regimes.py` o `tests/memory/test_regime_routing.py` con la copia sin mutar
en verde (60 pruebas):

- En la regla: la media en lugar de la mediana en los rendimientos diarios y en la tendencia,
  la tendencia como suma de las medianas diarias, los dos empates, el signo de la tendencia, el
  intercambio de las volatilidades, una ventana reciente con un rendimiento de más, los dos
  mínimos inclusivos, el calendario con otro módulo, otro mes o la hora de China, leer las
  sesiones ausentes, no comprobar presencias compartidas ni la sesión de la decisión, una sola
  ruta para los dos mercados y un calendario que clasifica cohortes pequeñas.
- En las claves: el bloque de Kronecker desplazado, el one-hot en un único compartimento y
  aceptar rutas fuera de rango.
- En la ventana: escribir con la ruta del evento de maduración en lugar de la de la decisión,
  no contar las escrituras e invertir el recuento de cambios.
- En la sesión: rutas desalineadas al liquidar y al emitir, la ruta de otro día y aceptar
  rutas pendientes fuera de rango.

No sobrevivió ninguna, así que no hubo mutantes equivalentes que justificar.
