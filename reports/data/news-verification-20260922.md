# Registro editorial del corpus

La cola contiene los **4.469.917 registros** del índice completo. No se ha
recortado el universo para crearla. Hay 3.049.555 registros estadounidenses y
1.420.362 chinos. Estos recuentos incluyen fuentes incompletas y registros que
no se pueden admitir para entrenamiento.

La inicialización de la política actual conserva exactamente las 45 revisiones
anteriores. Ninguna cambia de estado, texto, fecha o evidencia.

| Estado tras crear la cola | Registros |
| --- | ---: |
| Pendientes de contraste | 2.428.035 |
| Procedencia sin resolver | 1.061.815 |
| Periodo de test reservado | 980.008 |
| Cuerpo verificado anteriormente | 21 |
| Rechazados anteriormente | 11 |
| No verificables en revisiones anteriores | 13 |
| Sin contenido | 14 |

La [medición de inicialización](news-queue-init-v2-20260922.json) registra
75,70 segundos de operación, 121,78 MiB de RSS máximo y 650.170.368 bytes para
la base creada. El proceso completo incluye arranque y consultas de comprobación
adicionales, recogidas por separado en el
[registro de recursos](../resources/news-queue-init-v2-20260922-process-time.txt).
La ejecución coincidió con comprobaciones locales y no controló la caché del
sistema. No es una estimación del tiempo de entrenamiento.

La [primera pasada de la política actual](news-queue-pass-v2-20260922.json)
confirma 1.000 decisiones en 6,03 segundos de operación y 120,91 MiB de RSS
máximo. En 999 registros no hay todavía un editor compatible resuelto. Una
consulta devuelve 404 y conserva su plazo de reintento. No se añaden noticias
verificadas en esta pasada. Es un corte operativo para comprobar recuperación,
no un límite de empresas ni una selección para entrenar.

## Comprobaciones

La batería completa pasa **922 pruebas sin omisiones**. Ocho mutaciones dirigidas
se detectan en unidades, publicidad, identidad de bytes, integridad de capturas,
permisos de consulta, cuerpo editorial, protección del origen y conservación del
universo. No constituyen una puntuación exhaustiva de mutación.

El [recibo de calidad](../resources/news-verification-quality.json) conserva
herramientas, cobertura por módulo, complejidad y convención de CRAP. Las pruebas
incluyen reanudación, importación de revisiones, aislamiento por empresa,
reserva del test, respuestas truncadas y almacenamiento transaccional. El canal
de transporte controlado no reproduce todas las averías de una red real.

La comprobación con capturas actuales también encuentra diferencias respecto
a registros ya revisados, incluidas separación de palabras, puntuación y pies
de imagen. El contraste automático no borra esas diferencias para aumentar
la admisión. Las revisiones anteriores se conservan como evidencia de su propio
procedimiento, no como resultados nuevos del verificador.

## Trabajo pendiente

El adaptador implementado cubre estructuras observadas de The Motley Fool, no
todos los editores del corpus. Las fuentes sin resolver permanecen en la cola.
La procedencia y los cuerpos completos de las noticias chinas todavía necesitan
recuperación. También faltan sus publicaciones contables acreditadas y su factor
de mercado.

El entrenamiento completo de US, CN y US+CN no se ha ejecutado. El test final
sigue reservado y MARS-TITAN no se ha entrenado. La cola y sus comprobaciones
preparan ese trabajo, pero no sustituyen la admisión multimodal ni la evaluación
de los modelos.

Las órdenes, criterios y límites del registro se describen en la
[documentación de verificación](../../docs/data/news-verification.md).
