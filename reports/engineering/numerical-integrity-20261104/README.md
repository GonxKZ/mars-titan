# Integridad numérica y puntos de control

La revisión del 4 de noviembre corrige la pérdida de componentes complejas en las entradas,
el borrado de gradientes finitos grandes en el recorte de memoria y la subestimación del
espacio de los puntos de control. Las comprobaciones de CPU declaran su plataforma y el
informe nativo incorpora la proyección episódica y los medoids.

Los complejos se rechazan también dentro de arrays `object`, registros estructurados y
contenedores NumPy anidados. Los controles mantienen la conversión de valores reales,
incluidos `Decimal`, `Fraction` y registros de un campo. El recorte escala cada fila por
el máximo entre G y su mayor componente absoluta, y comprueba su precisión. Las filas que
ya cumplen mantienen sus valores. Se verifican derivadas de primer y segundo orden fuera
del umbral, donde la función es diferenciable.

El guardado cuenta los almacenamientos completos una sola vez. Conserva vistas y alias,
incluidos índices dispersos y parámetros de cuantización. Un escritor acota además los
bytes reales del temporal, aunque sus metadatos superen la estimación. La regresión incluye
40.000 vistas vacías y comprueba la conservación del archivo confirmado, la reserva de
16 MiB y el límite serializado. Un control conserva byte a byte el archivo habitual de
`torch.save`. El límite usa el espacio libre inicial y no reserva espacio frente a otros
procesos.

## Comprobaciones ejecutadas

El [recibo](verification.json) recoge las fases y sus límites. La conciliación de las pasadas,
las selecciones focales y sus reintentos da 18.654 identificadores
únicos superados y 438 omisiones por la protección del aprendizaje.
La selección focal de CPU pasa 188 casos y las paridades externas pasan 36. Se verifican
cuatro casos CUDA recogidos y una comprobación explícita adicional de paridad, gradientes y
cotas de PT1 en GPU. Los casos repetidos no se suman otra vez al total.
Los 94 identificadores únicos antes afectados por NVML pasan con una biblioteca oficial
compatible cargada solo para la comprobación. El controlador del sistema no se modifica.

Se detectan 34 [mutaciones dirigidas](mutations.json), con controles positivos antes
y después. Este número describe la selección realizada y no la tasa de mutación de todo
el proyecto. El control nativo distingue fallos de comportamiento, diagnósticos del
compilador y presencia en el informe de cobertura.

La recogida raíz contiene 20.930 casos. Quedan fuera de la selección ejecutada
1.792 casos con marca CUDA y 46 casos de ajuste sin
una entrada protegida. No se cuentan como superados. La batería CUDA general sigue
pendiente de su ventana de recursos y de las restricciones de aprendizaje que correspondan.

Ruff, formato y la comprobación del repositorio pasan. La compilación C++20 con clang-tidy,
análisis de rutas y Lifetime Safety pasa. ASan, UBSan y detección de fugas pasan cinco casos.
TSan pasa concurrencia y proyección. Las ocho funciones de prueba C++ modificadas tienen
230 líneas ejecutables cubiertas de 230 según LLVM, con CRAP máximo 7. Las métricas Python
figuran en [funciones modificadas](changed-functions.json). Se usa coverage.py 7.16.1 y
Radon 6.0.1. CRAP se calcula como C² × (1 − cobertura de sentencias)³ + C.

LLVM informa de dos mapeos sin perfil de `CashMovements`, con hash cero, al combinar los
perfiles del simulador y los medoids. El control aislado de medoids no los emite. Se
conservan ambos registros. La cobertura y CRAP son diagnósticos, no una demostración de
ausencia de defectos.

Desde la raíz, la comprobación focal de CPU se reproduce con:

```bash
uv run --offline --frozen --no-sync pytest \
  tests/models/test_baseline_complex_inputs.py \
  tests/models/test_baseline_object_inputs.py \
  tests/models/test_baseline_inputs.py \
  tests/models/titans/test_clip_rows_extremes.py \
  tests/training/test_checkpoint_storage_budget.py \
  tests/training/test_checkpoint_write_budget.py -m 'not cuda' -q
```

## Coste de la actualización de memoria

La [medida técnica](cost.json) compara la actualización de memoria completa con entradas
residentes y parámetros compartidos congelados. Se usan FP32, tres calentamientos y
quince repeticiones, sin TF32 en cuda:0. No mide el predictor completo. El equipo conserva
cargas concurrentes y los percentiles altos tienen pocas repeticiones.

| Dispositivo | Lote / dimensión / tokens | p50 anterior, ms | p50 corregido, ms | Diferencia absoluta máxima |
| --- | --- | --- | --- | --- |
| cpu | 4 / 16 / 4 | 1.635 | 1.759 | 3.73e-09 |
| cpu | 16 / 64 / 8 | 9.071 | 10.110 | 2.79e-09 |
| cpu | 32 / 128 / 8 | 59.977 | 73.163 | 1.86e-09 |
| cuda:0 | 4 / 16 / 4 | 5.110 | 5.237 | 4.66e-09 |
| cuda:0 | 16 / 64 / 8 | 9.863 | 10.401 | 2.79e-09 |
| cuda:0 | 32 / 128 / 8 | 10.245 | 10.394 | 2.33e-09 |

Los p95, p99 y picos del asignador CUDA están en el recibo de coste. Los resultados de
referencia se retiran de GPU antes de medir la otra implementación. El RSS registrado es
el pico de todo el proceso. No hay una diferencia aislada de RAM por implementación ni una
medida de energía. Estos casos no respaldan una aceleración general.

## Experimentos y requisitos pendientes

No se ejecutan entrenamientos ni pasos de optimizador y no se utiliza el test final
reservado. Las [omisiones por caso](remaining-cases.json) conservan su motivo. No hay mejora
predictiva medida. La revisión de dependencias se sigue en
[#832](https://github.com/GonxKZ/mars-titan/issues/832).
