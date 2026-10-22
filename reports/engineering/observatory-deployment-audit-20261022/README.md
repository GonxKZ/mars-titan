# Despliegue pendiente del observatorio

La auditoría del 22 de octubre de 2026 no permite cerrar #451. El código de main `8d8096b109262f4a6d848acd3b7f48c40fcac289` ya incluye las campañas por ventanas, pero el servicio utiliza el runtime limpio `05462aedc0290ed0ebedbad1db0415a7336001e5`, del 6 de octubre. El script de entrada coincide entre ambos. Los módulos de recogida y publicación difieren y falta `window_campaigns.py` en el runtime activo.

Hay tres diferencias que resolver:

- La configuración efectiva declara 88 fuentes y ninguna `window_campaign`. La versión de main tiene 30 fuentes e incluye las ocho nuevas. Sustituir el archivo efectivo dejaría 66 fuentes actuales fuera. Conservarlas y añadir las ocho daría 96, dentro del límite de 128.
- No existen los ocho resúmenes declarados de A y A v2. La búsqueda acotada no acreditó otro root equivalente. El recibo de vistas del 14 de octubre, SHA-256 `b0e708868fe5c923b9f4bc13870f7a2f8143013403f8df165e87dd2a5d7944c1`, declara `training_executed=false` y `final_test_opened=false`. No acredita ejecuciones de las cuatro etapas.
- Pages y el checkout de datos sirven el mismo índice, SHA-256 `33e743a87e601592bff3af79d4cd3e40179c6f246d2a768db72e7c8684fe87de`, con 3.319 registros, 88 campañas y ninguna matriz por ventanas. Su revisión de datos es `25447833a25792a7a7e962fb5a9eb2d14a62fdb8`, del 17 de octubre. El frontend se empaquetó el 20 de octubre. La rama mantiene 7.773 páginas para 51 referenciadas.

El colector está activo. En los índices inspeccionados sólo cambia `generated_at`, un campo que el código excluye de la identidad de publicación. La fecha antigua del índice público no prueba por sí sola una interrupción del servicio.

Se observó un escritor local y su cerrojo por directorio de estado. No se ha probado la exclusión de dos publicadores con estados distintos y el mismo checkout. Se propone un relevo sin solape y, como mitigación técnica pendiente, un cerrojo compartido por checkout con una prueba aislada de concurrencia. Esa mitigación no está implementada ni validada por esta auditoría.

Faltan el runtime actualizado, la configuración compatible, la identificación o ausencia explícita de las ocho fuentes y el recibo del ensayo sin publicar y del relevo posterior. No se deben crear resúmenes ficticios ni sustituir vistas preparadas por resultados de campaña.

Las [huellas de fuentes e índices, las fechas y el alcance de lectura](public-receipt.json) acompañan este informe. La [guía del observatorio](../../../docs/engineering/observatory.md#despliegue-sin-cortar-la-publicación-actual) describe el contrato y el procedimiento de relevo. La auditoría se limitó a lectura. No se ejecutaron colectores, entrenamiento ni cargas GPU y no se alteraron servicios o la rama de datos.
