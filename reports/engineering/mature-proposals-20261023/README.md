# Revisión madura y registro de propuestas

Este incremento de [#711](https://github.com/GonxKZ/mars-titan/issues/711) convierte
emisiones y resultados anteriores a un corte en una señal de revisión y conserva
propuestas limitadas a configuraciones registradas. Su
[contrato](../../../docs/engineering/mature-proposals.md) separa propuesta,
selección y despliegue. El selector aprendido y el ciclo exterior no están
implementados por estas funciones.

La suite conjunta pasó 80 casos sin errores u omisiones, 27 del registro anterior
y 53 de revisión y persistencia. Las emisiones de prueba se generan con fixtures,
el enlace nativo existente y `cuda:0`. El proceso no aplica pasos de optimizador.
La revisión y el registro se comprueban además con espías que prohíben construir
predictores, hacer forwards o ajustar parámetros.

Se verifican la mediana exacta de cinco cuantiles emitidos, la recepción nativa de
etiquetas, los dos anclajes esperados, pérdidas desconocidas, ganancias negativas,
cuotas, idempotencia, concurrencia de escritores y recuperación. Una regresión
reproduce el bloqueo de una apertura FIFO y comprueba su rechazo con `O_NONBLOCK`
mediante un proceso de metadatos nuevo. Otra impide aceptar un recuento maduro que
contradiga las filas. La validación conserva el snapshot nulo permitido para un
banco vacío.

Coverage 7.16.1 y Radon 6.0.1 registran 547/563 sentencias y 105/114 ramas en los tres
archivos de propuestas, con CRAP máximo 24. `mature_review.py` cubre 266/277
sentencias y 48/54 ramas. `ledger.py` cubre 188/193 y 29/32 respectivamente. El
registro previo cubre 93/93 y 28/28. CRAP usa `v²(1−c)³+v`, con complejidad Radon y
cobertura de sentencias en el rango de función. Las ramas se informan aparte.
Estas cifras no demuestran ausencia de defectos.

Dos mutaciones dirigidas completaron original G1 aprobado, fallo esperado en CALL
y restauración G2 aprobada. Una elimina la huella esperada de los eventos y otra
relaja la conciliación entre cargos y recibos. Sus seis fases no tuvieron errores
de colección ni omisiones. Duraron 48,936 s en total, con fuentes originales,
binario nativo y protección global intactos. No se cuentan como pruebas nuevas
ni estiman resistencia universal a mutaciones.

El [resumen verificable](summary.json) contiene fuentes, huellas de ejecuciones,
alcances y cobertura. No hay una evaluación financiera, un selector entrenado,
una medida de rendimiento del ciclo exterior ni una autenticación de archivos
coherentemente falsificados. El test final y la reanudación automática siguen
cerrados.
