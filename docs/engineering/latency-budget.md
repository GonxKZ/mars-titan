# Presupuesto de latencia, RAM y VRAM

MARS-TITAN busca reducir el error bajo límites de tiempo y memoria en un equipo con 32 GB de RAM y RTX 4070 Max-Q de 8 GB. Todavía no existen mediciones de latencia del modelo. Los límites siguientes son propuestas de diseño que el piloto deberá aceptar o revisar antes de la comparación confirmatoria. El perfil térmico y de potencia del portátil forma parte de la medida. No se extrapola el rendimiento de una RTX 4070 de sobremesa.

## Qué tiempo se medirá

| Medida | Inicio y final | Qué impide ocultar |
| --- | --- | --- |
| Latencia de cálculo en GPU | Eventos de dispositivo alrededor de la operación medida. | Confundir el envío asíncrono con la terminación del cálculo. |
| Latencia de predicción disponible | Entrada ya validada y transformada hasta salida numérica utilizable. | Transferencias, consulta de memoria, bucles y calibración. |
| Latencia completa | Recepción del evento hasta predicción validada. | Lectura, tokenización, extracción de embeddings, colas y serialización. |
| Antigüedad del estado | Diferencia entre corte de la instantánea y decisión atendida. | Una salida rápida calculada con un estado demasiado antiguo. |
| Tiempo de actualización | Etiqueta madura hasta estado nuevo publicable. | Consolidación aplazada que nunca alcanza al flujo. |
| Coste total del estudio | Preparación, búsqueda, entrenamiento, calibración y evaluación. | Desplazar todo el gasto al modelo maestro de destilación o a una caché que luego se omite. |

Se informarán p50, p95, p99, máximo, tamaño de muestra, caudal y proporción de incumplimientos. Un promedio pequeño no describe las colas de latencia. Los tiempos en frío, calentamiento, compilación y ejecución estable se publicarán por separado. Las [herramientas de medida de PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) son la referencia para temporización, junto con una medida de reloj de pared del recorrido completo.

## Presupuestos de trabajo propuestos

| Recurso | Presupuesto inicial | Condición |
| --- | --- | --- |
| RAM de procesos del proyecto | 16 GiB de suma PSS como objetivo, además de registrar RSS y caché del sistema. | Revisar disponibilidad real y medir todos los trabajadores, no solo el padre. |
| VRAM del proyecto | Objetivo de hasta 6 GiB de memoria reservada por el proceso principal. | Incluir contextos externos y memoria visible en `nvidia-smi`. No equivale al consumo total del equipo. |
| Cola de lotes fijados | Hasta 1 GiB dentro del presupuesto de RAM. | La cifra es un límite inicial, no una recomendación de ocuparlo. |
| Recurrencia de inferencia | K dentro de {1, 2, 4}. | Pesos compartidos, memoria congelada durante la consulta y techo de tiempo. |
| Latencia disponible, lote de un activo | Objetivo exploratorio p99 de 5 ms para el predictor compacto con entradas ya preparadas. | No se presenta como medido ni incluye un codificador de noticias grande. |
| Cohorte completa de un mercado | Objetivo exploratorio de 1 s con todas las predicciones disponibles. | Informar número de activos y presión concurrente. No extrapolar del lote de uno. |

Los valores sirven para hacer falsable la restricción de eficiencia. Se comparará la frontera completa aunque un objetivo aspiracional no se alcance. FinMultiTime diario no aporta los datos de órdenes necesarios para declarar competitividad en negociación de alta frecuencia. Su estudio requiere otra definición de tarea y otras fuentes.

## Modelo de coste y recurrencia

Para una consulta, una descomposición de diseño es:

$$L = L_{cola}+L_{lectura}+L_{transferencia}+L_{codificacion}+K L_{refinamiento}+L_{cabeza}+L_{calibracion}.$$

No todos los términos son constantes ni se solapan sin competir. Compartir pesos puede reducir almacenamiento, pero K pasos siguen repitiendo operaciones. La puerta de parada también tiene coste. Se compararán K fijo y puertas adaptativas a igual información y mediante una frontera de error frente a tiempo, no únicamente a igualdad de parámetros.

Un límite duro K = 4 no garantiza por sí solo una latencia máxima de extremo a extremo si la cola crece. Se acotan la cola, el tiempo de espera y la antigüedad aceptable de la instantánea. Superar un límite devuelve un estado explícito de abstención o de predicción no disponible. No se inventa una salida para cumplir el cronómetro.

## Memoria del modelo y sus actualizaciones

El presupuesto contiene pesos, estados, activaciones, gradientes, optimizador, cachés, tensores temporales y margen del asignador. La inferencia y el entrenamiento tienen perfiles distintos. No se estima la VRAM multiplicando solo parámetros por dos bytes.

Una actualización con doble buffer puede mantener dos juegos de estados o parámetros. Debe incluirse ese pico. Con una sola GPU, consolidar en segundo plano puede aumentar p99 aunque el proceso use otro stream. Se medirán tres escenarios: inferencia aislada, inferencia con carga de datos y consolidación simultánea, e inferencia bajo ráfagas de publicaciones. Si la coexistencia incumple el presupuesto, se programará la consolidación fuera de la ventana de predicción.

La [semántica CUDA de PyTorch](https://docs.pytorch.org/docs/2.14/notes/cuda.html) guía las dependencias entre streams y la sincronización. La [guía de NVIDIA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) orienta la reducción de transferencias y el acceso a memoria. Usar más streams no multiplica los recursos físicos.

## Orden de las optimizaciones

Primero se medirá la referencia correcta con precisión suficiente. Después se estudiarán lotes y bloques adecuados, caché de representaciones, reducción de copias y precisión mixta. Compilación, formas estáticas y captura de grafos se incorporarán solo cuando el perfil permita valorar su coste de preparación y sus restricciones.

La cuantización y los kernels propios se evaluarán después de comprobar estabilidad de estado, error de regresión y calibración. No se acepta una mejora de tiempo que cambie silenciosamente el objetivo o elimine el intervalo de incertidumbre. FlashAttention, kernels de estado y librerías CUDA tienen compatibilidades concretas. Se consultarán sus versiones y se medirá en esta RTX, sin trasladar cifras de A100 o H100.

## Protocolo de medida

Conservar hardware, versión de software, modo energético, temperatura, carga ajena, número de hilos, tamaños, precisión, estado de compilación y revisión del modelo. Repetir secuencias representativas y alternar el orden de variantes para reducir efectos de caché o temperatura. Evitar sincronizaciones dentro de cada operación si no forman parte del recorrido real, pero sincronizar correctamente los extremos que se cronometran.

El informe incluirá tanto valores del asignador como observación externa de memoria y procesos. Si un sensor de energía no está disponible, se declara. Los TFLOPS anunciados del dispositivo no son una medida de energía ni de latencia del modelo.

Una variante solo se promueve cuando sus intervalos de error, cobertura y tiempos permiten justificar el cambio. La latencia mínima absoluta, la precisión máxima y la memoria mínima pueden corresponder a soluciones distintas. La decisión se hará sobre restricciones explícitas y resultados medidos.
