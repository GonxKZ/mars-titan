# Checkpoints y recuperación de ejecuciones largas

El equipo puede permanecer encendido 24 horas al día. Las futuras ejecuciones se diseñan para detenerse y continuar sin reconstruir todo el trabajo. Este documento fija la política y las pruebas necesarias. Todavía no hay un entrenador ni checkpoints de pesos de MARS-TITAN.

## Qué se guarda

Un checkpoint corresponde a un `run_id`, un fold y una barrera temporal coherente. Incluirá:

- Pesos, optimizador, scheduler, escalador de precisión mixta y acumulación de gradientes cuando corresponda.
- Estado de generadores aleatorios de Python, NumPy, CPU y `cuda:0`, más las semillas de muestreo.
- Época, paso de optimización, bloques completados, posición de lectura confirmada y orden del sampler.
- Memoria global, estados por activo, episodios, versiones de representación y registros necesarios para restaurar la consulta.
- Cola de etiquetas pendientes, eventos maduros ya aplicados y predicciones originales que permiten calcular sus errores.
- Normalizadores, calibrador, política de recurrencia, configuración resuelta y límites de recursos.
- Hashes de datos, bloques, embeddings, código, dependencias, esquema de estado y revisión de reglas macro y mercado.

Los datos y las representaciones grandes se referencian mediante rutas lógicas y hashes. No se copian los 109 GiB de fuentes a cada checkpoint. El artefacto debe contener lo necesario para reconstruir su estado, y detectar si alguna dependencia ya no coincide.

## Barrera coherente

La política inicial guarda después de un paso de optimización completo y de cerrar la cohorte de predicciones y escrituras que corresponda. Ningún checkpoint puede mezclar pesos nuevos con optimizador antiguo, memoria parcialmente actualizada o un cursor que ya haya avanzado más que sus predicciones confirmadas.

Se prefiere guardar sin acumulación de gradientes pendiente. Si se admite otro punto, deben conservarse esos gradientes y el contador exacto. Las operaciones CUDA necesarias se sincronizan antes de copiar el estado. Una señal de interrupción solicita guardado en la siguiente barrera segura, no serializa tensores que todavía cambian.

La prelectura de un trabajador no equivale a consumo confirmado. El cursor persistido representa muestras cuyos efectos están comprometidos. Los elementos adelantados en colas se reconstruyen a partir de identificadores y orden deterministas. Si no puede recuperarse exactamente ese orden, se reinicia desde la última barrera anterior y se comprueba que no se duplican efectos.

## Frecuencia y escritura

La [configuración de trabajo](../../configs/checkpoints.toml) propone una solicitud cada 15 minutos, al terminar una época, antes de cambiar de fold y ante interrupción controlada. El guardado se materializa en la siguiente barrera segura. Si un paso tarda más de 15 minutos, esa periodicidad no es una cota del trabajo que se puede perder.

Se escribe en una ubicación temporal del mismo volumen, se comprueban tamaños y hashes y se sincronizan archivos. El manifiesto de confirmación se publica al final mediante una operación atómica del sistema de archivos. La restauración ignora directorios incompletos. La implementación deberá comprobar la durabilidad de ese protocolo y el comportamiento del sistema de archivos, sin asumir que un simple nombre `latest` basta.

Se conservan al menos los dos últimos checkpoints completos y el mejor según validación. Las referencias de cierre de fold se mantienen para reproducibilidad. La retención solo actúa sobre artefactos identificados del mismo `run_id` y después de validar el reemplazo. Nunca elimina datos fuente ni un mejor modelo usado por otro informe. El espacio se estima antes de comenzar una ejecución.

## Restauración segura

El cargador comprobará esquema, hash, revisión del código, configuración, fold, datos y compatibilidad de representación antes de reservar recursos. Los parámetros se cargarán desde un formato de tensores o un estado restringido a tipos seguros. No se deserializan objetos arbitrarios de checkpoints descargados. Toda migración entre esquemas es una operación explícita y verificable.

Se seleccionará `cuda:0` de nuevo y se comprobará su disponibilidad. Una GPU ausente produce un error que permite conservar el checkpoint. No cambia automáticamente a CPU ni inicia otra ejecución como si fuera la original.

Una actualización de software o controlador puede impedir igualdad bit a bit. Se distinguirá recuperación exacta dentro del entorno fijado de recuperación con tolerancia numérica documentada. Cambiar la versión de un codificador sin reconstruir sus memorias es incompatible aunque las dimensiones coincidan.

## Pruebas de aceptación

Se compararán una ejecución continua y otra interrumpida y restaurada en la misma barrera, examinando las siguientes predicciones, pérdidas, pasos, eventos aplicados y estados. Se probará interrupción durante escritura, manifiesto ausente, archivo truncado, hash incorrecto, cursor inconsistente, versión incompatible y falta temporal de GPU.

También se comprobará reanudación con etiquetas aún pendientes y con una publicación macro en cola. Ningún reinicio puede hacer que un evento se conozca antes de su disponibilidad. La recuperación de un fold no reutiliza el estado futuro de otro.

Un checkpoint se considera utilizable cuando pasa una carga de prueba y permite continuar. La mera existencia del archivo no acredita esa propiedad.
