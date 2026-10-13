# Preprocesado ajustado solo con el pasado en la publicación (13 de octubre de 2026)

Este informe acompaña al cuarto bloque de [#442](https://github.com/GonxKZ/mars-titan/issues/442). La issue pide que normalizaciones, residualizadores, umbrales de régimen y estratos de liquidez se ajusten solo con el pasado de cada corte, y que una prueba que introduce a propósito un ajuste con datos futuros haga fallar la comprobación. Recoge qué comprueba la sección `preprocessing` del informe de integridad que bloquea la publicación, cómo se ha comprobado que detecta esos ajustes y cuánto cuesta. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto.

## Dónde se ajusta algo con datos

| Paso | Dónde se calcula | Qué estima con datos | Comprobación |
| --- | --- | --- | --- |
| Etiquetas residuales (alfa y beta frente al factor de mercado) | Edición v3.1, fuera de la campaña (`data/residual_arrays.py`, referencia `data/budget_targets.py`) | Alfa y beta con la historia anterior a cada decisión | Auditoría del código y código declarado por la edición |
| Calendario de regímenes | `evaluation/regime_calendar.py`, una vez antes de evaluar | Rutas por volatilidad y tendencia con ventanas que terminan en cada decisión | Auditoría del código y código declarado por el calendario |
| Ridge y XGBoost | Ajuste de cada ventana | Medias, escalas y cortes de histograma | `fitted_rows` igual a las filas de entrenamiento |
| MARS-TITAN con M3 | Ajuste de cada ventana | Escalas de escritura del banco | Índice, edición y decisiones de entrenamiento, con etiquetas maduras antes de validación |
| Padres de la etapa de adaptadores | Etapa de adaptadores | Normalización de las entradas | Tramo y filas de entrenamiento, con etiquetas maduras antes de validación |
| Referencias neuronales, Titans-MAC, GRU episódica y CM-v1 | Ajuste de cada ventana | Ninguna estadística | Inventario con el motivo, y una familia nueva falla |
| Secciones de la comparación (calibración, estratos de modalidad, cuartiles, retención, estratos de liquidez) | Comparación walk-forward | Calibración en su tramo, el resto sin umbrales ajustados | Inventario de reglas causales |

Las comparaciones de A y A v2 declaran también estratos de liquidez. Su regla aplica umbrales fijos de precio y volumen a la edición sin ajustar hasta la última sesión publicada al emitir, y `tests/evaluation/test_liquidity_strata.py` comprueba que las filas posteriores no cambian el estrato. Una sección que no esté en el inventario hace fallar el informe hasta que se declare su regla.

## Auditoría del código

Una transformación que solo mira al pasado da el mismo valor en una decisión aunque cambien los precios posteriores. `past_only` recalcula con los precios posteriores a tres cortes (45, 70 y 90 % de la historia) multiplicados alternativamente por 1,7 y 0,6, y compara bit a bit:
- en las dos implementaciones del residualizador, alfa, beta, pares de historia y motivo de cada decisión hasta el corte, y el objetivo y su fecha de disponibilidad de las filas que ya habían madurado;
- en las rutas de régimen, la ruta, los activos y los rendimientos de cada sesión hasta el corte.

Los precios son deterministas y se construyen en el propio módulo (un activo y su factor de Estados Unidos entre 2021 y 2023, y 30 activos durante 200 sesiones para el régimen). No alimentan ningún modelo. Una auditoría que no compara ninguna fila no pasa.

## Edición y calendario

Las etiquetas y el calendario se calculan fuera de la campaña, así que la auditoría solo vale si la campaña usa el mismo código. El informe sigue cada comparación walk-forward publicada hasta sus vistas y los manifiestos padre de sus contratos temporales. Cada padre debe tener la huella que fijan sus vistas y declarar en su configuración el código de las etiquetas, que debe coincidir con el auditado. El calendario de regímenes debe declarar `code_sha256` igual al de `regime_calendar`. Con `real_edition_only` (A v2) la falta de código declarado o de ediciones también falla. Las campañas reducidas de las pruebas, con etiquetas escritas a mano, solo lo informan.

El manifiesto de las etiquetas v3.1 (`targets-v3.1/manifest.json`, SHA-256 `fe86a923dd49a24247a1925e195b0bfd07cacf02c7bd43f0bcde0d292251893a`) declara estas huellas, que coinciden con el código de esta rama:

| Campo | Archivo | SHA-256 |
| --- | --- | --- |
| `target_implementation_sha256` | `data/residual_arrays.py` | `d590c8074a98aaf5637a239faedbda0f94c46757a844a5bbe01add90db639f15` |
| `target_reference_sha256` | `data/budget_targets.py` | `6dfe6f353e4a5de5f69c7bacec9172c6a1b41b01b6f73e8a8022583c9fd8cb1d` |
| `temporal_reference_sha256` | `data/temporal.py` | `d860c1ac8e3e2e53773a51cc86464b9c9063830278c9ab15a136e3679b722d70` |

El [diagnóstico de los residuos](../../targets/diagnostics.md) ya volvió a calcular desde los precios, con esas mismas funciones, las 18.377.972 etiquetas aceptadas de la v3.1, y coinciden bit a bit en objetivo y momento de maduración. El calendario real de A v2 todavía no se ha construido.

## Pruebas con ajustes deliberados con el futuro

`tests/integrity/test_past_only_preprocessing.py` introduce cada ajuste a propósito y espera que la comprobación falle:
- un residualizador con la beta de toda la muestra y otro que divide el objetivo por la desviación de todos los objetivos;
- rutas de régimen con la ventana desplazada cinco sesiones hacia delante y con el umbral de turbulencia en la mediana de toda la serie;
- unas estadísticas tabulares que recorren también validación, escalas M3 de otro índice, de otra edición, de otras decisiones o con etiquetas que maduran en validación, y una normalización de un padre con otro tramo, otras filas o etiquetas tardías;
- una edición que no es la de sus vistas o con otro código, un calendario de otro código y código sin declarar con la política real;
- una sección sin inventariar y reglas no causales en los estratos de liquidez, la presencia de modalidades, la calibración, los regímenes y la cartera.

`tests/integrity/test_publication_gate.py` comprueba que la sección bloquea la publicación y que el calendario de regímenes llega al informe.

## Mutación dirigida

Se aplicaron 34 mutantes dirigidos, uno a uno, y cada uno ejecuta las pruebas que deberían detectarlo con `pytest -x` ([registro](mutation.json)). Mueren los 34. Al añadir los estratos de liquidez al inventario se repitieron los dos mutantes de reglas con las pruebas ampliadas, y el registro guarda esa segunda ejecución:

| Parte | Mutantes | Qué rompen |
| --- | --- | --- |
| Auditoría del código | 7 | Una auditoría sin filas comparadas que pasa, parámetros del residualizador o objetivos maduros sin comparar, precios futuros sin alterar, alteración antes del corte, rutas sin comparar y auditoría siempre superada |
| Edición y calendario | 6 | Una edición que no es la de sus vistas, etiquetas de otro código, código sin declarar o ninguna edición con la política real, y un calendario de otro código o sin código sin informar |
| Ajustes de la campaña | 9 | Filas de ajuste sin cotejar, escalas M3 de otro índice, de otra edición, de otras decisiones o con etiquetas tardías, escalas fuera de M3, M3 sin escalas, familia desconocida y recibos sin confirmar leídos |
| Etapa de adaptadores | 6 | Normalización de otro tramo, con más filas o con etiquetas tardías, ajustes cronológicos leídos, padre repetido y sección de la etapa ausente |
| Reglas y sección | 3 | Sección sin inventariar, regla no causal y política real ignorada |
| Conexión con la publicación | 3 | Informe sin la sección, informe sin el calendario y publicación que no pasa el calendario al informe |

Las pruebas que los detectan (`tests/integrity/test_past_only_preprocessing.py` y `tests/integrity/test_publication_gate.py`) corren con la protección de aprendizaje vigente y no rebajan el bloqueo.

## Coste

`measure_audit_cost.py` mide la auditoría del código, que es lo único que la sección calcula. El resto lee recibos, informes y manifiestos que el informe ya lee en otras secciones. Con dos hilos y la CPU compartida con otras cargas (carga media de 18 a 24), cinco repeticiones alternas ([medida](audit-cost.json)):

| Parte | Filas comparadas | Mediana | Mínimo y máximo |
| --- | --- | --- | --- |
| Residualizador de referencia (`budget_targets`) | 8.514 | 0,95 s | 0,89 a 1,07 s |
| Residualizador de la edición (`residual_arrays`) | 8.514 | 0,20 s | 0,18 a 0,21 s |
| Rutas de régimen | 1.239 | 0,11 s | 0,09 a 0,11 s |
| `audit_code` completa | | 1,20 s | 1,14 a 1,42 s |

El pico de memoria del proceso fue de 0,57 GiB. La auditoría se repite una vez por publicación, así que su coste es despreciable frente al recálculo de las métricas. La historia del residualizador tiene 753 sesiones (de 2021 a 2023) y la del régimen 200 sesiones con 30 activos.

## Límites

- La auditoría usa precios construidos para la prueba, no la edición. Demuestra que el código no mira al futuro con esa historia, y la edición queda ligada a ese código por las huellas declaradas.
- Las estadísticas de Ridge, XGBoost y M3 se comprueban por los recuentos, índices y fechas que declaran sus informes. El informe no vuelve a calcular las medias ni los cortes de histograma.
- La campaña no se ha ejecutado. No hay ninguna cifra de resultados.
