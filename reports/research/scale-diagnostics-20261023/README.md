# Fase 0: escala, ordenación y contracción de las predicciones de los pilotos

Este diagnóstico abre el estudio de variantes de [#742](https://github.com/GonxKZ/mars-titan/issues/742) y responde a [#743](https://github.com/GonxKZ/mars-titan/issues/743). Pregunta si los predictores de los pilotos de octubre ordenan los residuos y pierden frente a cero por exceso de amplitud, o si no hay señal de ordenación que aprovechar. No ajusta ningún modelo, no importa PyTorch y no abre el test de 2024. El año 2023 es desarrollo reutilizado.

## Resultado

**Con la familia declarada, ningún predictor de la población principal tiene un IC medio distinguible de cero.** La [regla declarada](../../../configs/evaluation/scale-diagnostics-phase0.json) antes de calcular lleva entonces a usar el IC como contraste principal de la criba de #742.

**Esa conclusión depende de cómo se defina la familia.** La declaración fija la corrección «dentro de la familia de la población», con los 281 predictores que tienen IC en todas las sesiones. Si la familia se limita a otros conjuntos, aparece un IC positivo en cada caso. Ninguno resiste las dos sensibilidades de bloque.

| Familia | Tamaño | IC positivo con intervalo simultáneo | Bloques de 8 y 32 días |
| --- | ---: | --- | --- |
| Población completa (la declarada) | 281 | Ninguno | |
| Solo la fuente principal (pérdida puntual) | 19 | `both_isolated-46`, IC 0,0198 [0,0008, 0,0388] | Deja de excluir el cero con 32 |
| Los seis estudios con inventario | 56 | `cascade_original-42`, IC 0,0158 [0,00005, 0,0316] | Deja de excluir el cero con 8 |

La lectura honesta es que hay, como mucho, una señal de ordenación débil y frágil. No hay evidencia de que no exista, ni de que sea estable. Las dos sensibilidades de familia se añadieron tras la revisión de la PR y no cambian la decisión declarada.

La amplitud explica la pérdida frente a cero. El λ elegido con la partición de calibración tiene una mediana entre 0 y 0,019 según el estudio, y vale exactamente cero en el 43 % de los predictores de la población principal y en el 61 % de los de 256 activos. En la calibración, la pendiente mediana es 0,015 en la población principal y 0,006 con 256 activos. Con esas pendientes, las predicciones tendrían entre 65 y 165 veces la amplitud que les corresponde. En 2023 la pendiente mediana de la población principal es −0,006, negativa en 152 de los 281 predictores con pendiente definida. La relación que se observa en calibración no se mantiene en la evaluación.

Tras contraer con λ de calibración, el exceso mediano frente a cero es del 0 % porque la predicción queda casi en cero. La mejor mejora frente a cero es del 0,028 % en la población principal y del 0,053 % con 256 activos. Con λ elegido sobre la propia evaluación (una cota con información futura, no un resultado) llega al 0,064 % y al 0,080 %. Frente al efecto mínimo detectable de su propia serie de diferencias, estas mejoras quedan cortas:

| Población | Mejora | Frente a cero | Efecto detectable / mejora (8, 26, 32 candidatas) |
| --- | --- | ---: | --- |
| Principal | λ de calibración | 0,028 % | 9,4 / 10,5 / 10,7 |
| Principal | Cota con información futura | 0,064 % | 1,9 / 2,2 / 2,2 |
| 256 activos | λ de calibración | 0,053 % | 1,3 / 1,5 / 1,5 |
| 256 activos | Cota con información futura | 0,080 % | 2,1 / 2,4 / 2,4 |

Batir a cero en MAE no es imposible, pero el efecto detectable es entre 1,3 y 11 veces mayor que la mejor mejora observada. Como el número de sesiones necesario crece con el cuadrado de esa razón, harían falta entre unas 1,7 y 115 veces más sesiones que las 249 de 2023. Con 256 activos la distancia es menor.

## Datos

- Hay 757 carpetas de predicciones. Se analizan 754, que dan 522 predictores únicos.
  - 232 carpetas repiten exactamente las predicciones de otro predictor y se cuentan una sola vez. Sus alias están en `summary.json`. Un cero con signo negativo se trata como cero.
  - Se omiten tres carpetas del estudio combinado: las celdas congeladas `cell0_frozen-42` y `cell7_frozen-42`, que solo tienen evaluación, y `cell0_m3_zero_refinement-42`, que no tiene predicciones. Esta regla de omisión se añadió después de la declaración. En un estudio con inventario publicado, una partición ausente es un error.
- Seis estudios entran a través de sus inventarios publicados. La huella SHA-256 de cada archivo se comprueba antes de leerlo, y la de cada inventario queda en `summary.json`.
  - [Controles de la pérdida puntual](../point-loss-controls-20261020/development-v1/README.md)
  - [Cuantiles con cinco semillas](../quantile-multiseed-20261020/development-v1/README.md)
  - [Enrutamiento de cuantiles](../quantile-routing-20261020/development-v1/README.md)
  - [Captura local](../local-capture-20261020/development-v1/README.md)
  - [Plasticidad v2](../neural-plasticity-20261020/development-v2/README.md)
  - [Competencia de claves](../key-competition-20261020/development-v1/README.md)
- El [estudio combinado](../combined-memory-20261018/README.md) y las cuatro poblaciones v4 de la [ronda de refinamientos](../refinement-development-20261019/completed-v1/README.md) se leen por su estructura de carpetas. La huella de cada archivo queda registrada.
- Partición que elige λ: calibración, de octubre a diciembre de 2022 (62 sesiones).
- Evaluación: 2023 (249 sesiones).
- Sensibilidad: λ elegido con validación y calibración juntas. Solo existe en los 56 predictores de los estudios con inventario y en los 42 del estudio combinado, porque las carpetas de refinamientos no guardan validación. La validación de abril a septiembre de 2022 sirvió para seleccionar en esos pilotos, así que es optimista.

Hay tres poblaciones de evaluación distintas, cada una con su propia familia de contrastes:

| Población | Filas | Activos | MAE de cero | Predictores |
| --- | ---: | ---: | ---: | ---: |
| Principal, 64 activos de EE. UU. | 15.936 | 64 | 0,01584945 | 282 |
| Estudio combinado, 64 activos con otras etiquetas | 15.936 | 64 | 0,01231802 | 42 |
| Refinamientos con 256 activos | 63.679 | 256 | 0,01333346 | 198 |

El estudio combinado usó etiquetas distintas de las de los pilotos posteriores. Por eso su MAE de cero es otro y no se mezcla con la población principal.

## Método

| Medida | Cálculo |
| --- | --- |
| λ por mercado | Menor λ en [0, 1] que minimiza el MAE por sesión de λ·x en la calibración. Se resuelve de forma exacta como mediana de y/x ponderada con \|x\|/n de cada sesión. Se aplica sin cambios a la evaluación |
| Pendiente de calibración | Σw·x·y / Σw·x², con el mismo peso por sesión, en la partición de calibración y en la evaluación |
| IC | Correlación de Pearson por sesión, media en 2023. También se informa el IC de rangos. La familia del IC excluye los predictores con alguna sesión sin correlación definida. Solo hay uno en la población principal y otro en la de 256 activos, los controles que predicen exactamente cero |
| Control de mediana pasada | Mediana de las últimas 60 etiquetas del mismo activo con dos sesiones de retraso, desde la validación. Con menos de 20 etiquetas predice cero |
| Intervalos | Bootstrap circular por bloques de 16 días UTC, 2.000 réplicas y semilla 20261023. Intervalo simultáneo por máximo estudentizado dentro de cada población, con sensibilidad a bloques de 8 y 32 |
| Efecto mínimo detectable | Bonferroni unilateral con α = 0,05, potencia 0,8 y tamaño efectivo AR(1) de la serie diaria de diferencias frente a cero. Es optimista frente al intervalo simultáneo bilateral que decide |

Las pruebas de [`scale_diagnostics`](../../../src/mars_titan/evaluation/scale_diagnostics.py) y del [script](../../../scripts/diagnose_prediction_scale.py) cubren:

- λ frente a una rejilla densa, con amplitud conocida, empates y recorte
- pesos por sesión en λ y en la pendiente
- el retraso de la mediana pasada y su alineación con la evaluación
- que λ sale solo de la calibración y la sensibilidad de validación y calibración
- que la serie contraída usa el λ de calibración
- el rechazo de un ajuste posterior o simultáneo a la evaluación
- los bloques circulares y el recorte de la autocorrelación
- la decisión de la regla, su sensibilidad a la familia y la mejor mejora frente a cero
- la huella de los inventarios, el cero con signo y las carpetas omitidas.

## Población principal por estudio

Los porcentajes son excesos de MAE frente a predecir cero. «IC > 0» e «IC < 0» cuentan intervalos marginales al 95 % que excluyen el cero. El percentil 90 de λ se toma por orden.

| Estudio | Predictores | Exceso en bruto, mediana (mín.–máx.) | IC medio, mediana (máx.) | IC > 0 / IC < 0 | λ mediano (p90) | λ = 0 | Exceso tras λ, mediana (mín.) |
| --- | ---: | --- | --- | ---: | --- | ---: | --- |
| Pérdida puntual | 19 | 9,18 % (0,38–85,98) | 0,0002 (0,0224) | 3 / 1 | 0,014 (0,135) | 7 | 0,000 % (−0,008) |
| Cuantiles con semillas | 10 | 29,74 % (1,77–91,88) | −0,0040 (0,0122) | 1 / 1 | 0 (0,050) | 6 | 0,000 % (0,000) |
| Captura local | 15 | 52,84 % (17,47–175,57) | −0,0015 (0,0105) | 0 / 0 | 0 (0,018) | 9 | 0,000 % (0,000) |
| Plasticidad v2 | 7 | 38,92 % (6,33–60,71) | 0,0058 (0,0158) | 1 / 1 | 0 (0,012) | 4 | 0,000 % (−0,009) |
| Competencia de claves | 5 | 49,99 % (8,83–54,65) | 0,0029 (0,0140) | 0 / 0 | 0,017 (0,024) | 2 | 0,000 % (−0,009) |
| Refinamientos, original | 103 | 5,47 % (0,00–342,87) | 0,0060 (0,0235) | 6 / 0 | 0,019 (0,219) | 34 | 0,005 % (−0,028) |
| Refinamientos, geometría | 123 | 6,51 % (0,56–98,89) | 0,0011 (0,0164) | 2 / 3 | 0,003 (0,128) | 58 | 0,000 % (−0,004) |

El enrutamiento de cuantiles no aparece como fila porque sus cinco predictores repiten exactamente otros del estudio de cuantiles con semillas. El mínimo de 0,00 % en refinamientos es el control que predice exactamente cero.

Detalles de la tabla:

- **IC.** De los 281 intervalos marginales, 13 son positivos y 6 negativos. El azar daría unos 7 por lado, así que el exceso de positivos es pequeño y los predictores están muy correlacionados entre sí.
- **Contracción.** Ningún predictor contraído mejora a cero con el intervalo simultáneo. Tres lo hacen con el marginal y 51 empeoran de forma significativa, porque el λ de 62 sesiones no se sostiene en 2023.
- **Mediana pasada.** Pierde frente a cero un 0,80 %, con un intervalo simultáneo que excluye el cero.

## Otras poblaciones

**Estudio combinado.**

- La mediana del IC medio es −0,0064 y el mejor es `mac_frozen` con 0,0302.
- Hay 7 intervalos marginales positivos y 1 negativo, ninguno simultáneo.
- Ningún predictor contraído supera a cero.

**Refinamientos con 256 activos.**

- Hay 25 intervalos marginales positivos y ninguno negativo entre 197 predictores.
- Uno tiene un intervalo simultáneo positivo, el control `threshold` de `linear_v1-43` con lector K4 y gráficos originales: IC 0,0167 [0,0008, 0,0325]. Se mantiene con bloques de 8 días y deja de excluir el cero con bloques de 32.
- Tres predictores contraídos superan a cero con el intervalo simultáneo, pero tienen λ entre 1,5·10⁻⁶ y 1,9·10⁻⁴ y mejoras inferiores al 0,0002 %. En la práctica predicen cero.
- La mediana pasada pierde un 1,02 % frente a cero.

La regla se declaró sobre la población principal, así que estos resultados no la cambian. Se informan porque apuntan a que una sección transversal más amplia aporta precisión al IC.

## Potencia

| Población | Candidatas | Efecto detectable (% del MAE de cero) | Autocorrelación mediana |
| --- | ---: | ---: | ---: |
| Principal | 8 / 26 / 32 | 1,43 / 1,60 / 1,63 | 0,18 |
| Estudio combinado | 8 / 26 / 32 | 1,38 / 1,54 / 1,57 | 0,14 |
| 256 activos | 8 / 26 / 32 | 1,10 / 1,23 / 1,25 | 0,15 |

Estas cifras usan las diferencias de las predicciones en bruto, cuya dispersión es mucho mayor que la de las contraídas. Para una predicción contraída hay que usar su propia serie, como en la tabla de la sección de resultado.

## Consecuencias para la criba

- Por la regla declarada, el contraste principal pasa a ser el IC medio por sesión de cada variante frente a su control. El MAE calibrado y el MAE frente a cero se mantienen como secundarios declarados. El protocolo de la criba debe recoger también que el veredicto depende de la familia.
- La mejor contracción con 256 activos se queda a 1,3-1,5 veces del efecto detectable, frente a 9-11 veces con 64 activos. Junto con el único IC simultáneo positivo, esto apoya cribar con 256 activos.
- Contraer hacia cero es necesario para no perder frente a cero, pero en estos pilotos equivale casi siempre a apagar el modelo. No es una mejora de predicción y no se presentará como tal.

## Limitaciones

- 2023 se ha usado ya en todos estos pilotos. El diagnóstico describe esos predictores y no estima su rendimiento en datos nuevos.
- La calibración tiene 62 sesiones, poco para fijar λ. En los 56 predictores de la población principal que guardan validación, el λ mediano es 0,0015 con solo la calibración y cero con validación y calibración juntas.
- Muchos predictores comparten código, padre o semilla. El intervalo simultáneo tiene en cuenta su correlación en el bootstrap, pero el número de familias independientes es menor que el de predictores.
- Solo hay datos de EE. UU. en estos pilotos. China queda pendiente de la criba.

## Reproducción y coste

```bash
uv run python scripts/diagnose_prediction_scale.py \
    --declaration configs/evaluation/scale-diagnostics-phase0.json \
    --root <estado privado de los pilotos> \
    --output reports/research/scale-diagnostics-20261023
```

La ejecución publicada salió del commit `ca5c4ab6bbea3758fabe935a0d488dda98de2375` con el árbol limpio. Tardó 279 s de reloj y 317 s de proceso, con un máximo de 422 MB de RSS en CPU, sin GPU y con cero pasos de optimizador. El [resultado completo](summary.json) contiene cada predictor, sus huellas, alias, intervalos y sensibilidades. La primera versión de este informe contaba 544 predictores y afirmaba que la mejor mejora estaba veinte veces por debajo del efecto detectable. La revisión de la PR corrigió ambas cifras.
