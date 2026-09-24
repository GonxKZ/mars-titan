# Serialización de eventos en los checkpoints nativos

El ejecutable C++ conserva ahora el último evento como un `StepOutcome` propietario y lo convierte a JSON cuando publica un checkpoint. Antes construía ese JSON en cada transición, aunque el siguiente evento lo sustituyera sin llegar a guardarse. La contabilidad y el intervalo de confirmación permanecen iguales.

El perfil previo con Callgrind 3.26.0 registró 937.728.799 referencias de instrucciones en el proceso completo de nueve escenarios y 128 activos. La construcción y destrucción de objetos JSON y sus asignaciones figuraban entre los caminos con más instrucciones. Este perfil instrumentado localiza trabajo, pero sus porcentajes no representan tiempo de pared ni ancho de banda del hardware.

El [perfil posterior](native-checkpoint-profile.json) registra 646.559.763 referencias, un 31,1 % menos. Permanecen las operaciones de inicialización, hashing, serialización de los checkpoints confirmados y contabilidad. Se conservan las bibliotecas que las implementan. No se dispone de medidas de tráfico de memoria y FLOPs que permitan atribuir un límite Roofline ni justificar sustituir esas primitivas por kernels propios.

La consulta de `perf stat` no pudo abrir eventos porque el sistema utiliza `perf_event_paranoid=4`. No se cambiaron permisos ni ajustes del kernel. El perfil de instrucciones anterior utiliza Callgrind y no se presenta como un contador del hardware.

La [comparación de ejecutables](native-checkpoint-benchmark.json) utiliza procesos separados, orden alternado, un calentamiento y cinco repeticiones por caso. Se conserva el binario anterior y se prepara una cinta común para cada tamaño. La referencia y el candidato usan Clang 21.1.8 Release, las mismas bibliotecas y la misma política numérica.

| Activos y decisiones | Trabajadores | Referencia, ms | Serialización aplazada, ms | Cambio de la mediana |
| --- | ---: | ---: | ---: | ---: |
| 16 y 48 | 1 | 156,78 | 161,28 | +2,9 % |
| 16 y 48 | 8 | 62,35 | 60,87 | −2,4 % |
| 128 y 192 | 1 | 330,46 | 319,46 | −3,3 % |
| 128 y 192 | 8 | 115,08 | 110,35 | −4,1 % |
| 512 y 256 | 1 | 539,29 | 430,45 | −20,2 % |
| 512 y 256 | 8 | 183,75 | 157,56 | −14,3 % |

Son tiempos del proceso completo, incluidos arranque, lectura, cálculo y persistencia de los nueve escenarios. Se conserva el resultado desfavorable de la carga pequeña con un trabajador. Sus rangos se solapan, entre 153,89 y 172,28 ms en la referencia y entre 151,78 y 166,78 ms en el candidato. No se acredita una mejora de ese caso. La mejora más clara corresponde a 512 activos, donde hay más eventos por serializar. Las diferencias de RSS medianas son inferiores a 0,3 MiB y no se presentan como ahorro de memoria.

Las 60 ejecuciones cronometradas conservan exactamente las métricas, el snapshot y el último evento confirmado, comprobados mediante huellas. La pausa en un paso intermedio también compara su evento con otra ejecución que confirma todos los pasos antes de reanudar. Una excepción conserva el checkpoint anterior confirmado. El cambio añade como máximo un evento pendiente y no acumula un historial en RAM.

La última compilación pasa clang-tidy y el analizador de rutas de Clang sobre las cinco fuentes de producción. ASan/UBSan pasa cuatro CTests y 75 pruebas del ejecutable. TSan pasa cuatro CTests y cinco casos concurrentes. El perfil de fuzzing pasa siete pruebas, incluidos tres programas de 1000 entradas. Una [mutación dirigida](native-checkpoint-mutation.json) compila y hace fallar la comprobación del evento guardado al pausar entre checkpoints. El [diagnóstico de cobertura](native-checkpoint-quality.json) registra 1943 de 2264 líneas y 979 de 1474 ramas. Se conserva el aviso de LLVM sobre las entradas inline duplicadas, explicado en la [verificación anterior](native-financial-verification.md).

El script [native_checkpoints.py](../../benchmarks/native_checkpoints.py) recibe `--reference` y `--candidate`, con las rutas de los dos ejecutables, `--work` con una carpeta nueva, `--output` y `--repetitions`. El informe incluye huellas de ambos binarios y sus fuentes, configuración, tamaños, repeticiones y resultados individuales. Las bibliotecas de cada binario deben conservarse junto a su compilación durante la comparación.

La caché del sistema permanece caliente y las aplicaciones existentes siguen en marcha. `VmHWM` mide el máximo residente desde `exec`, compartido entre hilos. No se mide GPU, energía ni calidad predictiva. Esta optimización se conserva por su beneficio en el recorrido grande y mantiene la referencia contable. No sustituye el estudio de entrenamiento acelerado.
