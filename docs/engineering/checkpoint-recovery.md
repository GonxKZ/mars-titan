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

## Implementación para las referencias

`training/checkpoints.py` implementa persistencia en Linux para los modelos de
referencia. Guarda un estado de tensores y tipos simples, junto con una identidad
de datos y configuración. El entrenador debe incluir pesos, optimizador, RNG,
época y cursor confirmado después de la actualización. La API no inventa memorias
adaptativas ni colas que estos modelos no utilizan.

La escritura mantiene un bloqueo exclusivo, sincroniza el archivo y publica
`latest.json` después del estado. Conserva dos estados recientes, el mejor marcado
por el consumidor y hasta 64 referencias fijadas explícitamente. El consumidor
debe fijar un estado antes de usarlo como origen de una continuación independiente.
La limpieza afecta únicamente a los archivos de estado identificados de esa
ejecución. El límite serializado y descomprimido es de 512 MiB.

La carga verifica huella, tamaño, esquema e identidad antes de usar
`torch.load(weights_only=True)`. Si el último archivo está truncado o cambia su
huella, intenta el anterior confirmado y emite un aviso. No busca archivos
huérfanos para convertirlos en estados válidos. Una identidad distinta impide
la carga y el guardado. La reserva de espacio previa no garantiza que otro
proceso no agote el disco durante la escritura.

El guardado comprueba también una carga restringida antes de publicar el
manifiesto. Un objeto serializable que el cargador no admita no reemplaza un
estado válido.

`StopRequest` convierte SIGINT y SIGTERM en una solicitud. La integración con
el entrenador debe atenderla en la siguiente barrera segura. No interrumpe una
actualización a medias ni promete recuperarse de SIGKILL guardando después de
recibirlo.

Las pruebas cubren un proceso terminado durante una escritura, errores de disco,
retención, corrupción y restauración de una GRU con su optimizador y generadores.
La siguiente actualización reproduce exactamente pesos y pérdida en CPU y en
`cuda:0`. Esto no acredita recuperación frente a un corte eléctrico ni persistencia
de los futuros módulos de memoria de MARS-TITAN.

## Época seleccionada y estado de reanudación

Las referencias pueden seleccionar una época mediante MAE por sesión. La
decisión se toma después de evaluar una época completa. Los empates conservan
la primera época aceptada. La paciencia cuenta épocas consecutivas sin una
mejora mayor que `min_delta`. No altera las filas de entrenamiento ni consulta
la reserva final.

`recovery_checkpoint` identifica el último estado coherente, con su optimizador,
cursor y generadores. `checkpoint` identifica el estado elegido para las
predicciones finales y las continuaciones. El índice conserva ese estado como
`best`. Si se interrumpe la generación de predicciones del modelo elegido, no
se vuelve a guardar mezclando esos pesos antiguos con el optimizador reciente.

La carga de un estado elegido comprueba que corresponda a la época y puntuación
de selección, con cursor vacío y estadísticas de entrenamiento reiniciadas.
Una referencia a otra época válida también es un error. Para una continuación
se exige la huella concreta del origen. En ese caso, el cargador no sustituye
el archivo solicitado por uno anterior. La recuperación operativa de `latest`
sin huella específica mantiene la alternativa anterior con aviso.

El parámetro opcional de un caso es:

```json
{
  "selection": {
    "metric": "session_mae",
    "patience": 5,
    "min_delta": 0.0
  }
}
```

`epochs` sigue siendo el máximo de épocas. `stopped_early` indica si el criterio
detuvo el ajuste antes de agotarlo. El informe conserva todas las épocas
ejecutadas, la época elegida y las dos referencias de estado. Los casos sin
`selection` mantienen el número fijo de épocas anterior.
