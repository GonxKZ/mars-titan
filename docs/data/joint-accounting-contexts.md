# Proyección de conceptos contables separados

`project_numeric_context` prepara un contexto común sin convertir monedas ni
declarar equivalencias contables. Recibe una columna Arrow
`FixedSizeList<float32>`, los identificadores del origen y los del destino.
Conserva el orden de bloques: valores, máscaras y edades. La salida es otra
columna del mismo tipo, con tres componentes por concepto de destino.

Cada concepto original debe aparecer exactamente una vez en el destino. Los
conceptos añadidos tienen valor, máscara y edad cero. Esos ceros indican que
el canal no se aplica, no una observación financiera nueva. Los valores
originales conservan sus bits, incluidos los ceros con signo.

La unión prevista de los 23 conceptos USD/CAD y los tres conceptos CAS/CNY
produce 26 conceptos y 78 componentes. El patrimonio con minoritarios de la
fuente china conserva su identificador propio. La función no lo renombra como
patrimonio atribuible a los accionistas ni calcula ratios entre monedas.

Se rechazan vocabularios repetidos, conceptos perdidos, filas o componentes
nulos, NaN, infinitos, máscaras distintas de cero o uno y edades negativas.
Un canal ausente debe tener valor y edad cero. Entrada y salida suman como
máximo 64 MiB, incluidos los buffers retenidos por una vista Arrow. La copia
usa bloques de hasta 1 MiB entre entrada y salida, con un máximo de 512 conceptos
y 1.024 fragmentos.

## Comprobación y coste

Pasan 44 pruebas y ocho mutaciones dirigidas. La cobertura medida comprende
56 sentencias y 30 ramas. CCN y CRAP máximos son 16 con la convención indicada
en el [recibo](../../reports/resources/joint-context-projection-20261007.json).
Estas medidas no garantizan ausencia de errores.

La comparación usa una referencia NumPy por columnas y canal, con validación
equivalente para los tipos medidos. Hay paridad bit a bit en seis cargas de
32, 8.192 y 100.000 filas. Para 100.000 filas, la mediana pasa de 68,71 a
18,48 ms en la proyección 23→26 y de 10,03 a 6,68 ms en 3→26. En los dos casos
pequeños de 3→26, la candidata tarda unos 3,4 y 39,4 μs más que la referencia.

El pico de proceso más alto es 173.469.696 bytes, incluidos imports y datos de
prueba. Se midió con una cuota de una CPU, límite de 1 GiB y CUDA desactivada,
mientras la campaña científica seguía activa. El perfil incluye tiempo de
NumPy dentro de los marcos Python. No demuestra un cuello de botella de
Python ni justifica un kernel C++ propio.

Se ha medido esta función con datos generados para pruebas. Todavía no se
atribuye una aceleración a la materialización completa o al entrenamiento.
