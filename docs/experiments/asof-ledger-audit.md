# Auditoría as-of del registro de etiquetas

`asof_ledger_audit` es la variante G3 del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#748](https://github.com/GonxKZ/mars-titan/issues/748)) y forma parte del banco de integridad ([#442](https://github.com/GonxKZ/mars-titan/issues/442)). No es un modelo ni una etapa de salida. Mide la validez del registro de etiquetas que leen las etapas, no su rendimiento, así que no tiene control de rendimiento. Está implementada en `src/mars_titan/integrity/asof_ledger_audit.py` y comprobada con registros sintéticos o escritos a mano.

## Qué audita

Las etapas de salida y las calibraciones en línea leen etiquetas que maduran con el tiempo: la volatilidad pasada de A2, PT2 ([#454](https://github.com/GonxKZ/mars-titan/issues/454)) y su recalibración periódica, y A3 cuando se fusione ([#766](https://github.com/GonxKZ/mars-titan/pull/766)). El programa de salida de G1 no lee objetivos después de G, así que no tiene etiquetas que auditar. Cada una tiene sus propias pruebas de causalidad en un corte. Esta auditoría aplica la misma comprobación a cualquier etapa y en todos los cortes declarados, con un recibo común.

Un registro es una tabla de eventos con columnas alineadas. Cada evento tiene su instante de emisión (`issued_at`), su disponibilidad (`available_at`) y sus valores, como el objetivo residual. Siguiendo [Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html) (Fowler, 2005), lo que una etapa emite en t debe poder reconstruirse con los eventos disponibles en t. En términos de [W3C PROV-DM](https://www.w3.org/TR/prov-dm/) (Moreau y Missier, 2013), cada emisión solo puede derivarse de entidades que ya existían cuando se generó.

## Comprobaciones

`audit(emit, ledger, columns=..., cutoffs=..., stored=None)` recibe la etapa como una función del registro que devuelve sus emisiones y el instante de cada una.

1. **Registro válido.** Antes de auditar rechaza columnas desalineadas, valores no reales, eventos disponibles en el instante de su emisión o antes y cualquier instante de 2024.
2. **Reconstrucción.** Para cada corte t, las filas emitidas como mucho en t deben ser idénticas bit a bit con el registro original y con otro en el que los eventos disponibles después de t tienen otros valores. Se prueban dos sustituciones, cero y $\pm 10^6$ con el signo contrario al original. Una etiqueta disponible justo en t puede leerse.
3. **Repetición.** Dos ejecuciones sobre el mismo registro emiten lo mismo, la etapa no modifica el registro y, si se aporta una emisión guardada, la reproduce bit a bit.

Un fallo no lanza error. Queda en el recibo con `passed=False`, el corte y la sustitución, para que la puerta de integridad decida. El recibo guarda además, por corte, las filas emitidas, los eventos futuros y la huella SHA-256 de lo emitido hasta ese corte.

`rejection(emit, ledger)` comprueba otra cosa: que una etapa que recibe los dos instantes rechace por sí misma un evento disponible en el instante de su emisión o en 2024, en lugar de emitir con él.

## Hallazgos

La auditoría pasó la reconstrucción en todos los cortes en las tres etapas existentes. La comprobación de rechazo encontró dos registros imposibles que se aceptaban sin error. Ninguno cambiaba emisiones anteriores, pero ambos dejaban pasar datos que deberían invalidar la entrada:

- **PT2 en línea.** `replay_online_conformal` solo comprobaba la madurez máxima de cada cohorte. Una fila cuya etiqueta decía estar disponible en el instante de su predicción pasaba si otra fila de la cohorte maduraba después. Ahora cada fila debe madurar después de su predicción.
- **Recalibración periódica.** `replay_periodic_refit` aceptaba una etiqueta con madurez en 2024. Nunca llegaba a usarse, porque las emisiones son anteriores, pero PT2 ya cerraba 2024 y la alternativa trivial debe hacer lo mismo. Ahora se rechaza.

En la volatilidad pasada de A2 el historial no lleva el instante de emisión de cada etiqueta, así que la etapa no puede rechazar una madurez imposible por sí misma. La comprobación queda en el registro que la auditoría valida antes de ejecutarla. El lector de la criba deberá conservar ese instante ([#746](https://github.com/GonxKZ/mars-titan/issues/746)).

## Detección comprobada

Cada etapa se audita también en una versión con una fuga deliberada, que lee las etiquetas un día antes de su disponibilidad. La auditoría falla en las tres, solo por reconstrucción. Las pruebas del propio auditor usan una media de etiquetas disponibles, que pasa, y la misma media sobre todas las etiquetas, que falla en todos los cortes con eventos futuros y con las dos sustituciones. Leer una etiqueta un microsegundo antes de su disponibilidad falla exactamente en el corte afectado.

## Correspondencia con pruebas

Las pruebas están en `tests/integrity/test_asof_ledger_audit.py`.

| Comprobación | Código | Prueba |
| --- | --- | --- |
| Registro válido | `checked_ledger` | `test_impossible_or_malformed_ledgers_are_rejected_before_auditing` |
| Reconstrucción en etapas reales | `audit` | `test_each_stage_reconstructs_its_past_emissions_at_every_cutoff` |
| Detección de fugas | `audit` | `test_a_stage_reading_labels_one_day_early_is_detected`, `test_reading_every_label_is_detected_at_each_cutoff_with_future_events`, `test_a_label_read_exactly_when_it_becomes_available_is_allowed` y `test_a_leak_through_a_one_sided_transform_is_still_detected` |
| Repetición, solo lectura y emisión guardada | `audit` | `test_repetition_read_only_and_stored_emissions_are_checked` |
| Recibo | `audit` | `test_a_correct_stage_passes_and_its_receipt_is_reproducible` |
| Rechazo en la etapa | `rejection` | `test_stages_that_receive_both_instants_reject_impossible_records` |

## Límites

- La auditoría comprueba la causalidad respecto a las columnas declaradas. Si una etapa leyera el futuro a través de otra columna, como los cuantiles, habría que declararla también.
- Sustituir valores no cubre una etapa que dependa solo de la presencia de eventos futuros y no de su valor. Las etapas actuales reciben filas alineadas y no cambian de forma con la madurez.
- Mide validez técnica sobre registros sintéticos. Aplicarla a los Parquet de cada trabajo de la criba queda en #746, igual que sumar sus recibos al informe de integridad.

## Coste medido

La auditoría no tiene tensores. Su coste es el de repetir la etapa $2+2C$ veces con C cortes, más las copias del registro. `benchmarks/asof_ledger_audit_cost.py` lo mide sobre PT2 en línea, la etapa existente más cara, con 256 activos sintéticos de semilla fija y un corte por sesión. El recibo está en `reports/engineering/output-stages-20261023/asof-ledger-audit-cost.json`. Se tomó antes de rebasar la rama sobre `develop`, así que su campo `commit` no aparece en la historia publicada. Las huellas SHA-256 de las fuentes que guarda coinciden con las de esta versión. La máquina tenía una carga media de 41 durante la medida, así que estas cifras no son cotas superiores.

| Tramo auditado | Filas | Cortes | Emisiones | Una emisión de PT2 | Auditoría | Rechazo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 sesiones | 5.376 | 21 | 44 | 0,022 s | 0,75 s | 0,056 s |
| 63 sesiones | 16.128 | 63 | 128 | 0,028 s | 5,21 s | 0,071 s |

La auditoría del primer caso es la mediana de tres repeticiones y la del segundo una sola. El coste crece aproximadamente con el producto de cortes y filas, porque cada emisión recorre todo el tramo. Un año completo no se ha medido. Si hiciera falta acotarlo, se puede auditar una muestra de cortes declarada antes de mirar los resultados. No se midieron la GPU, la lectura de Parquet ni la energía.

## Mutaciones dirigidas

Se aplicaron a mano 15 mutaciones al auditor y se revirtieron después. Todas hacen fallar alguna prueba salvo una equivalente: quitar el salto de los cortes sin eventos futuros o sin emisiones, que solo ahorra ejecuciones. Una sobrevivía al principio: sustituir los eventos futuros por $+10^6$ sin invertir el signo. Motivó `test_a_leak_through_a_one_sided_transform_is_still_detected`, donde una fuga a través de $\min(y,0)$ con etiquetas positivas solo se ve con el extremo de signo contrario.

Las pruebas nuevas de PT2 (`test_a_row_whose_label_is_available_at_its_own_prediction_is_rejected`) y de la recalibración periódica (el caso `2024` de `test_declarations_and_rows_that_break_the_contract_are_rejected`) fallaban antes de sus correcciones y pasan después.
