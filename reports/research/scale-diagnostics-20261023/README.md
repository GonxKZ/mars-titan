# Fase 0: escala, ordenación y contracción de las predicciones de los pilotos

Este diagnóstico abre el estudio de variantes de [#742](https://github.com/GonxKZ/mars-titan/issues/742) y responde a [#743](https://github.com/GonxKZ/mars-titan/issues/743). Pregunta si los predictores de los pilotos de octubre ordenan los residuos y pierden frente a cero por exceso de amplitud, o si no hay señal de ordenación que aprovechar. No ajusta ningún modelo, no importa PyTorch y no abre el test de 2024. El año 2023 es desarrollo reutilizado.

## Resultado

**Ningún predictor de la población principal tiene un IC medio distinguible de cero con la corrección por familia.** La [regla declarada](../../../configs/evaluation/scale-diagnostics-phase0.json) antes de calcular fija la consecuencia. La criba de #742 usará el IC como contraste principal, y batir a cero en MAE no es alcanzable con estas entradas y este tamaño de muestra.

La amplitud explica la pérdida frente a cero, pero no hay una señal que recuperar al corregirla. El λ elegido con la partición de calibración tiene una mediana entre 0 y 0,019 según el estudio, y vale exactamente cero en entre el 41 % y el 57 % de los predictores según la población. La pendiente de calibración tiene una mediana de 0,015 en la población principal y de 0,006 con 256 activos. Las predicciones tienen entre 65 y 160 veces la amplitud que les correspondería con su correlación. Tras contraer, el exceso de MAE frente a cero queda en torno al 0 % porque la predicción contraída es casi cero. Incluso eligiendo λ con la propia evaluación (una cota con información futura, no un resultado), la mejor mejora frente a cero es del 0,064 % en 64 activos y del 0,080 % en 256.

El efecto mínimo detectable con 249 sesiones está entre el 1,4 % y el 1,6 % del MAE de cero para 8 a 32 candidatas. Es unas veinte veces mayor que la mejor mejora posible observada. Con estas entradas, una criba que compare MAE frente a cero no tiene potencia.

## Datos

- 757 carpetas de predicciones, de las que se analizan 754 y salen 544 predictores únicos.
  - 210 carpetas repiten exactamente las predicciones de otro predictor y se cuentan una sola vez, con sus alias en `summary.json`.
  - Tres carpetas del estudio combinado no tienen partición de calibración y quedan omitidas: los dos modelos congelados y una ejecución sin predicciones.
- Seis estudios entran a través de sus inventarios publicados, con la huella SHA-256 de cada archivo comprobada antes de leerlo.
  - [Controles de la pérdida puntual](../point-loss-controls-20261020/development-v1/README.md)
  - [Cuantiles con cinco semillas](../quantile-multiseed-20261020/development-v1/README.md)
  - [Enrutamiento de cuantiles](../quantile-routing-20261020/development-v1/README.md)
  - [Captura local](../local-capture-20261020/development-v1/README.md)
  - [Plasticidad v2](../neural-plasticity-20261020/development-v2/README.md)
  - [Competencia de claves](../key-competition-20261020/development-v1/README.md)
- El [estudio combinado](../combined-memory-20261018/README.md) y las cuatro poblaciones v4 de la [ronda de refinamientos](../refinement-development-20261019/completed-v1/README.md) se leen por su estructura de carpetas. La huella de cada archivo queda registrada en `summary.json`.
- Partición que elige λ: calibración, de octubre a diciembre de 2022 (62 sesiones).
- Evaluación: 2023 (249 sesiones).
- Sensibilidad: λ elegido con validación y calibración juntas. La validación de abril a septiembre de 2022 sirvió para seleccionar en esos pilotos, así que es optimista.

Hay tres poblaciones de evaluación distintas, cada una con su propia familia de contrastes:

| Población | Filas | Activos | MAE de cero | Predictores |
| --- | ---: | ---: | ---: | ---: |
| Principal, 64 activos de EE. UU. | 15.936 | 64 | 0,01584945 | 293 |
| Estudio combinado, 64 activos con otras etiquetas | 15.936 | 64 | 0,01231802 | 42 |
| Refinamientos con 256 activos | 63.679 | 256 | 0,01333346 | 209 |

El estudio combinado usó etiquetas distintas de las de los pilotos posteriores, por eso su MAE de cero es otro y no se mezcla con la población principal.

## Método

| Medida | Cálculo |
| --- | --- |
| λ por mercado | Menor λ en [0, 1] que minimiza el MAE por sesión de λ·x en la calibración. Se resuelve de forma exacta como mediana de y/x ponderada con \|x\|/n de cada sesión. Se aplica sin cambios a la evaluación |
| Pendiente de calibración | Σw·x·y / Σw·x², con el mismo peso por sesión |
| IC | Correlación de Pearson por sesión, media en 2023. También se informa el IC de rangos |
| Control de mediana pasada | Mediana de las últimas 60 etiquetas del mismo activo con dos sesiones de retraso, desde la validación. Con menos de 20 etiquetas predice cero |
| Intervalos | Bootstrap circular por bloques de 16 días UTC, 2.000 réplicas y semilla 20261023, con intervalo simultáneo por máximo estudentizado dentro de cada población y sensibilidad con bloques de 8 y 32 |
| Efecto mínimo detectable | Bonferroni unilateral con α = 0,05, potencia 0,8 y tamaño efectivo AR(1) de la serie diaria de diferencias frente a cero |

Las pruebas de [`scale_diagnostics`](../../../src/mars_titan/evaluation/scale_diagnostics.py) comprueban:

- El λ exacto frente a una rejilla densa, una amplitud conocida, el empate y el recorte.
- El peso por sesión.
- Que la mediana pasada nunca lee la última etiqueta sin madurar.
- Que λ no cambia al alterar las etiquetas de evaluación.
- El rechazo de una partición de ajuste posterior a la evaluación.

## Población principal por estudio

Los porcentajes son excesos de MAE frente a predecir cero. «IC > 0» e «IC < 0» cuentan intervalos marginales al 95 % que excluyen el cero. Ninguno lo excluye con el intervalo simultáneo.

| Estudio | Predictores | Exceso en bruto, mediana (mín.–máx.) | IC medio, mediana (máx.) | IC > 0 / IC < 0 | λ mediano (p90) | λ = 0 | Exceso tras λ, mediana (mín.) |
| --- | ---: | --- | --- | ---: | --- | ---: | --- |
| Pérdida puntual | 19 | 9,18 % (0,38–85,98) | 0,0002 (0,0224) | 3 / 1 | 0,014 (0,135) | 7 | 0,000 % (−0,008) |
| Cuantiles con semillas | 10 | 29,74 % (1,77–91,88) | −0,0040 (0,0122) | 1 / 1 | 0 (0,050) | 6 | 0,000 % (0,000) |
| Captura local | 15 | 52,84 % (17,47–175,57) | −0,0015 (0,0105) | 0 / 0 | 0 (0,018) | 9 | 0,000 % (0,000) |
| Plasticidad v2 | 7 | 38,92 % (6,33–60,71) | 0,0058 (0,0158) | 1 / 1 | 0 (0,012) | 4 | 0,000 % (−0,009) |
| Competencia de claves | 5 | 49,99 % (8,83–54,65) | 0,0029 (0,0140) | 0 / 0 | 0,017 (0,024) | 2 | 0,000 % (−0,009) |
| Refinamientos, original | 108 | 5,42 % (0,00–342,87) | 0,0060 (0,0235) | 6 / 0 | 0,019 (0,219) | 34 | 0,004 % (−0,028) |
| Refinamientos, geometría | 129 | 6,40 % (0,00–98,89) | 0,0011 (0,0164) | 2 / 3 | 0,003 (0,128) | 58 | 0,000 % (−0,004) |

El enrutamiento de cuantiles no aparece como fila porque sus cinco predictores repiten exactamente otros del estudio de cuantiles con semillas.

Detalles de la tabla:

- **IC.** Hay 281 predictores con IC definido en todas las sesiones y el valor crítico simultáneo es 3,48. Los 13 intervalos marginales positivos y 6 negativos son compatibles con el azar, que daría unos 7 por lado.
- **Contracción.** Ningún predictor contraído mejora a cero con el intervalo simultáneo. 51 empeoran de forma significativa porque el λ de 62 sesiones no se sostiene en 2023.
- **Mediana pasada.** Pierde frente a cero un 0,80 %, con un intervalo simultáneo que excluye el cero.

## Otras poblaciones

**Estudio combinado.**

- La mediana del IC medio es −0,0064 y el mejor llega a 0,0302 (`mac_frozen`).
- Hay 7 intervalos marginales positivos y 1 negativo, ninguno simultáneo.
- Ningún predictor contraído supera a cero.

**Refinamientos con 256 activos.**

- El patrón cambia algo. Hay 25 intervalos marginales positivos y ninguno negativo entre 197 predictores.
- Uno, el control `threshold` de `linear_v1-43` con lector K4 y gráficos originales, tiene un intervalo simultáneo positivo: IC 0,0167 [0,0008, 0,0325].
- Es una señal de ordenación pequeña, coherente con que más activos por sesión dan IC más precisos.
- Tres predictores contraídos superan a cero con el intervalo simultáneo. Su λ es del orden de 10⁻⁴ y su mejora no llega al 0,0001 %, así que en la práctica predicen cero.

La regla se declaró sobre la población principal, así que estos resultados no la cambian. Se informan porque apuntan a que la sección transversal amplia aporta potencia.

## Potencia

| Población | Candidatas | Efecto detectable (% del MAE de cero) | En desviaciones típicas | Autocorrelación mediana |
| --- | ---: | ---: | ---: | ---: |
| Principal | 8 / 26 / 32 | 1,38 / 1,55 / 1,57 | 0,25 / 0,28 / 0,29 | 0,17 |
| Estudio combinado | 8 / 26 / 32 | 1,38 / 1,54 / 1,57 | 0,24 / 0,27 / 0,28 | 0,14 |
| 256 activos | 8 / 26 / 32 | 1,03 / 1,15 / 1,17 | 0,24 / 0,27 / 0,28 | 0,14 |

La serie de diferencias tras contraer es cero en la mitad de los predictores, así que su efecto detectable no es informativo y no se presenta.

## Consecuencias para la criba

- El contraste principal pasa a ser el IC medio por sesión de cada variante frente a su control. El MAE calibrado y el MAE frente a cero se mantienen como secundarios declarados. Este cambio lo decide la regla declarada antes de calcular y no altera las demás decisiones de #742.
- Una sección transversal mayor mejora la precisión del IC. Los datos de 256 activos son los únicos con un IC simultáneo positivo. Cribar con 64 activos deja poca potencia para efectos de IC del orden de 0,02, así que conviene reconsiderar el tamaño de la población antes de registrar el protocolo.
- Contraer hacia cero es necesario para no perder frente a cero, pero en estos pilotos equivale casi siempre a apagar el modelo. No es una mejora de predicción y no se presentará como tal.

## Limitaciones

- 2023 se ha usado ya en todos estos pilotos. El diagnóstico describe esos predictores y no estima su rendimiento en datos nuevos.
- La calibración tiene 62 sesiones, poco para fijar λ. Con validación y calibración juntas el λ mediano sigue siendo 0 en la población principal.
- Muchos predictores comparten código, padre o semilla. El intervalo simultáneo tiene en cuenta su correlación en el bootstrap, pero el número de familias independientes es menor que el de predictores.
- Solo hay datos de EE. UU. en estos pilotos. China queda pendiente de la criba.

## Reproducción y coste

```bash
uv run python scripts/diagnose_prediction_scale.py \
    --declaration configs/evaluation/scale-diagnostics-phase0.json \
    --root <estado privado de los pilotos> \
    --output reports/research/scale-diagnostics-20261023
```

La ejecución publicada salió del commit `d14b95ab15ff746ccbf59cae07ce477e1a8f6147` con el árbol limpio. Duró 171 s de reloj y 203 s de proceso, con un máximo de 377 MB de RSS en CPU, sin GPU y con cero pasos de optimizador. El [resultado completo](summary.json) contiene cada predictor, sus huellas, alias, intervalos y sensibilidades.
