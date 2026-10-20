# Canales de precios en la preparación de medidas

El materializador del benchmark reservaba cinco canales de precios aunque el dataset declarase seis. La edición vigente produjo un error de forma antes de guardar entradas. La reserva utiliza ahora `dataset.price_channels`, con el mismo contrato que el lector de entrenamiento. No se trunca el sexto canal ni se cambia el predictor del estudio en curso.

Pasaron 13 casos CPU, sin fallos ni omisiones. La regresión cubre ambas anchuras, todos los valores de cada canal y ausencia de objetivos en el archivo. Los tres mutantes dirigidos quedaron detectados, con referencia inicial y restaurada aprobadas. El [recibo](verification.json), las [mutaciones](mutations.json) y el [análisis de cobertura y CRAP](quality.json) conservan las fuentes y el alcance.

`materialize` cubre 43/45 sentencias y 22/26 ramas, con complejidad 22 y CRAP 22,04. La cobertura del módulo completo es menor, porque contiene también recorridos CUDA no ejecutados por esta selección CPU. La materialización real y el benchmark posterior necesitan sus propios recibos. Estas pruebas no contienen entrenamiento ni resultados financieros.
