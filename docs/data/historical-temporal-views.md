# Vistas temporales con máscaras históricas

La política `historical_masked_2000_v1` crea vistas de entrenamiento, validación, calibración y evaluación sobre el corpus supervisado histórico. Requiere adhesión explícita en el productor y en `CorpusDataset`. La política estricta sigue siendo el valor predeterminado y conserva su admisión macro completa.

Los protocolos `configs/evaluation/historical-masked-us-walk-forward.json` y `historical-masked-cn-walk-forward.json` copian la matriz reciente de cada mercado y cambian únicamente `train_start` a `2000-01-01`. Mantienen diez ventanas, los cortes de validación, calibración y evaluación, las semillas, el margen de una sesión y la reserva desde 2024. No sobrescriben los protocolos de las ediciones anteriores. El [protocolo anual v2](../research/walk-forward-2000.md) añade ventanas desde 2005 en US y desde 2011 en CN sin modificar estos archivos. La selección se declara exclusivamente sobre `validation`. Un contrato que declare calibración como selección se rechaza.

La vista histórica conserva los vectores macro, los catálogos y las máscaras del padre. No recibe otro panel macro ni un informe `complete-decisions`. Un bloque ausente no elimina una fila con precios y objetivo válidos. El lector exige que los valores observados sean finitos y tengan disponibilidad pasada, y que el relleno, las presencias y las máscaras por concepto sean coherentes. Las fechas reservadas se comprueban antes de decodificar sus vectores.

Cada fila original conserva su índice y una etiqueta o causa de exclusión por ventana. El calentamiento y la ausencia de objetivo mantienen sus motivos originales. `recover_annual_boundaries=True` es obligatorio para esta política. Recupera únicamente el objetivo y la maduración ya guardados del corte de 2022 a 2023. Si el intervalo cabe en un mismo tramo del fold nuevo, puede admitirse. Si cruza una frontera real, se purga. No se recuperan etiquetas nulas, no maduras, posteriores al corte o que requieran abrir 2024.

El contrato temporal histórico tiene versión 2 e incluye política, contrato de máscaras, protocolo, ventana, selección, recuperación anual y hash del padre. La representación de la vista debe coincidir con la de ese padre. Los manifiestos supervisados conservan la versión 3 y los archivos de muestras originales. Las vistas solo escriben etiquetas y metadatos en un destino nuevo. Una interrupción anterior a la publicación permite repetir el intento y una edición confirmada no se sustituye.

`prepare_temporal_corpus` recibe el padre, el protocolo, `None` para macro y admisión, y el destino, con `input_policy="historical_masked_2000_v1"` y `recover_annual_boundaries=True`. La CLI admite `--input-policy historical_masked_2000_v1 --recover-annual-boundaries` y omite `--macro` y `--admission` en esta política.

`prepare_joint_temporal_corpus` recibe un protocolo por mercado. Conserva los cierres y festivos propios de US y CN y une sus filas por fase, sin intersectar instantes UTC. Los recuentos de la unión son la suma de los recuentos locales, incluidos los tramos que no tengan objetivos válidos. El censo sigue incluyendo los candidatos sin precios. Los cursores y las cachés conservan los IDs y sus presencias.

Esta preparación no ejecuta modelos ni acredita que el corpus histórico real esté materializado. Los consumidores de aprendizaje necesitan su propia adaptación explícita a la política histórica. La reserva y el bloqueo previo al aprendizaje permanecen vigentes.

El [recibo de comprobación](../../reports/data/historical-temporal-views-20261008.json) recoge las 154 pruebas técnicas, las diez mutaciones detectadas, la paridad estricta y la revisión independiente. La cobertura y la complejidad se incluyen con su convención y sus límites.
