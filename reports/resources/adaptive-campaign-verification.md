# Verificación de la campaña y de su benchmark

El coordinador ejecuta 21 pilotos y elige la duración de las siete variantes principales a partir del tiempo consumido. Las tres semillas son 42, 43 y 44. La consolidación añade seis casos si se cumple el criterio de validación. La auditoría abre sus 512 mundos después de congelar las selecciones.

El presupuesto incluye 12 horas de piloto, 108 principales, 36 auxiliares y 12 de reserva. Una imputación inicial de ingeniería se registra separada del tiempo observado y se descuenta de la reserva y del total. Un intento fallido también consume presupuesto. La espera de admisión GPU no se cuenta como ejecución.

La salida 2 permite tres reanudaciones separadas por 30 segundos. Las salidas 1 y 75 bloquean el caso. Un recibo incoherente queda bloqueado antes de poder reanudarlo. El diario recupera cierres confirmados que ocurrieron antes de su última escritura, sin volver a entrenar esos casos.

El ejecutable configura en Linux la terminación al morir su supervisor. Se rechazaron dos fallos reproducidos durante la revisión: aceptar una identidad alterada con el sello anterior y atribuir una reserva finita a un hijo antiguo que pudo sobrevivir sin vigilancia. La prueba de cierre forzado conserva un proceso ajeno. Los estados anteriores sin ese vínculo de supervisión no se migran como si su duración estuviera acreditada.

La verificación inicial del coordinador y del lanzador pasó 70 pruebas. Coverage.py 7.16.2 midió 860 de 915 líneas ejecutables y 248 de 282 ramas. Radon 6.0.1 calculó un CRAP máximo de 41, con cobertura de líneas por función. Esa medición precede a la publicación del registro ligero.

La integración posterior de registro y benchmark pasó 91 pruebas. El registro inicial contiene un caso activo y veinte pendientes, todos sin transiciones confirmadas. No duplica el resumen del coordinador como si fuera otro entrenamiento. Las publicaciones de pausa y bloqueo conservan el trabajo confirmado.

El benchmark usa un calentamiento por variante, trabajador y binario. Alterna A/B y B/A, conserva las mismas fuentes y detiene la medida ante una salida incorrecta. Las pruebas comprueban entrada inmutable, archivos retenidos, exclusión del calentamiento y cierre de procesos propios. La referencia antigua se protege mediante un descriptor de proceso, abierto tras comprobar su relación con el supervisor.

Las pruebas instrumentadas del benchmark cubrieron 178 de 186 líneas ejecutables y 40 de 44 ramas. Su función principal concentra las comprobaciones de entrada y alcanzó un CRAP de 45,003. Las funciones añadidas para publicar el registro tienen todas sus líneas recorridas. La convención utilizada es CCN² × (1 − cobertura de líneas)³ + CCN. Estas cifras ayudan a localizar código pendiente de más casos.

Se detectaron seis mutaciones del coordinador y seis del benchmark. Alteraban sellos, presupuestos, reconciliación, selección auxiliar, orden de las medidas, integridad de entradas o cierre de hijos. Las pruebas de estos contratos utilizan recibos y procesos controlados. Las mediciones reales de entrenamiento se registran por separado.

El preflight sobre el catálogo preparado confirmó 256 fuentes de entrenamiento, 128 de validación y 512 entradas reservadas. Las primeras 16 fuentes de entrenamiento incluyen dos mundos de cada familia. Esa comprobación verificó el ajuste HMM de entrenamiento y no abrió archivos de auditoría.
