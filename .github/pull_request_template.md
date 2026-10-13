## Contexto

Issue que resuelve (`Closes #N` o `Refs #N`), objetivo de investigación o tarea transversal y motivo del cambio.

## Cambio

Comportamiento final y decisiones tomadas. Si cambia una identidad, una huella o un formato publicado, decir cuál y por qué.

## Criterios comprobables

- [ ] Cada criterio de la issue con la prueba, la orden o el recibo que lo comprueba.

## Evidencia

Resultados de pruebas, medidas o informes enlazados. Distinguir lo medido de lo esperado e indicar equipo, versiones y repeticiones cuando haya medidas.

## Comprobaciones

- [ ] `make check` pasa. Si se toca código nativo o CUDA, también `make test-native` o `make test-cuda`.
- [ ] Fechas de disponibilidad, particiones y prueba final revisadas si cambian datos, modelos o selección.
- [ ] Configuración, semillas y procedencia registradas si hay experimentos.
- [ ] No se incluyen datos restringidos, secretos, pesos, rutas personales ni PDF de terceros.
- [ ] Documentación actualizada y commits en inglés con Conventional Commits.

## Límites

Lo que queda fuera, los riesgos conocidos y lo que no se ha podido comprobar.
