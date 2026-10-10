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
| 64 | 0,47 ms | 3,08 ms | 0,30 ms / 2,65 ms | 1,2 % / 10,8 % |
| 128 | 0,57 ms | 3,53 ms | 0,40 ms / 3,08 ms | 0,8 % / 6,3 % |
| 1.024 | 3,16 ms | 10,58 ms | 2,38 ms / 8,83 ms | 0,6 % / 2,2 % |
| 4.096 | 11,40 ms | 46,67 ms | 8,96 ms / 40,86 ms | 0,6 % / 2,6 % |

La ruta cuesta lo mismo con las dos claves y crece con la cohorte (8,6 ms con 4.096 filas,
sobre todo la mediana de NumPy). Con `codec_by_regime` domina la escritura proximal con 320
coordenadas, una factorización de Cholesky de LAPACK de 320 × 320 más el producto KᵀWK.
Resolver solo los compartimentos con etiquetas la acercaría a la del codec, pero ahorraría
como mucho unas doce horas de GPU entre los dos brazos en A y no se ha implementado. La carga
media del equipo rondaba 22 por otros procesos durante la medida, registrada en el recibo. Una
primera medida del mismo día, con una carga parecida y antes de dar formato al script, dio
cifras entre un 15 % y un 50 % menores, así que la dispersión entre ejecuciones es grande.
No se midió en GPU porque la ruta y A trabajan en CPU y FP64, ni la energía, por falta de
instrumento.

## Mutación dirigida

[`mutations.json`](mutations.json) recoge 26 mutaciones aplicadas de una en una sobre una
copia aislada de `src`, `tests` y `configs`, con las pruebas importando la copia. Cada entrada
guarda el fragmento original y el mutado. Las 26 hicieron fallar
`tests/memory/test_regimes.py` o `tests/memory/test_regime_routing.py` con la copia sin mutar
en verde (59 pruebas):

- En la regla: la media en lugar de la mediana, los dos empates, el signo de la tendencia, el
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
