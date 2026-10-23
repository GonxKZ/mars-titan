# Verificación de la parada residual del puente

La [variante](../../../docs/experiments/adaptive-evidence-bridge-v1.md) está conectada al puente, a los consumidores financieros, a la recuperación nativa, a la colección de rutas y al registro de configuraciones. Permite contrastar la recurrencia fija, la referencia adaptativa densa y la compactación, con parámetros emparejados y estados nuevos. Permanece desactivada por defecto.

El código se conserva en los commits `91df7a63a`, `174e0d669` y `a3760c93f`. El [recibo legible por máquina](summary.json) incluye los sellos de las fuentes, comprobaciones y mutaciones. Los tiempos publicados corresponden a comprobaciones instrumentadas, sin acreditar latencia de inferencia ni ahorro de entrenamiento.

## Resultados técnicos

La integración del lector y la recuperación superó 391 casos. La regresión posterior de la colección, las variantes y la materialización superó 284. Hay 597 identificadores de prueba distintos entre esas dos pasadas, con solapes que impiden sumar sus tamaños. Ambas terminaron sin fallos ni omisiones. Incluyen rutas CPU y CUDA, FP32 y FP64. Las guardas registraron cero intentos y cero pasos de optimizador.

Se comprobaron valores y gradientes del prefijo seleccionado, independencia de RNG, máscaras, matrices nulas y casos extremos de FP64. Los casos de integración conservan IDs y parámetros, rechazan checkpoints incompatibles y reproducen el estado confirmado. La colección contrasta el trabajo declarado con las filas realmente evaluadas por la transición. La referencia densa paga J máximo aunque seleccione J1.

La revisión encontró un fallo en la recuperación del consumidor financiero. Declarar cero refinamientos y eliminar la traza podía evitar la conciliación. Se reprodujo con una prueba que falló antes de corregir el contrato. Ahora los refinamientos se contrastan con el tipo de evento nativo y la decisión requiere su traza.

Las mutaciones dirigidas alteraron la frontera inclusiva, la máscara anterior a la resta y la admisión del coste denso. Las tres se detectaron en CALL con referencias verdes antes y después. La comprobación del coste se repitió tras extraer su validador común, por lo que hay cuatro ejecuciones de mutación y tres defectos distintos. No es una tasa general de detección de mutantes.

## Cobertura y límites

Se usaron coverage 7.16.1 con ramas y Radon 6.0.1. CRAP se calcula como `v²·(1-c)³+v`, con complejidad ciclomática v y fracción de sentencias cubiertas c dentro del rango de la función. Las ramas se informan por separado. No se añadieron exclusiones de cobertura.

| Ámbito y pasada | Sentencias | Ramas | CRAP máximo |
| --- | ---: | ---: | ---: |
| Núcleo de recurrencia | 175/186 | 51/58 | 23 |
| Once fuentes de integración | 2.607/3.067 | 1.023/1.390 | 231,66 |
| Ocho fuentes de colección y registro | 1.171/1.330 | 388/544 | 177,23 |

Los ámbitos y las versiones de las fuentes difieren. El máximo de integración procede de una función anterior de memoria asociativa poco ejercitada en esa selección. El máximo de la tercera pasada pertenece al validador de la traza regional. Las cifras señalan límites de estas pruebas y no describen la cobertura de todo el repositorio ni demuestran ausencia de errores.

Las comprobaciones se ejecutaron localmente con una RTX 4070 Laptop de 8 GiB, CUDA explícita y un límite de 1 GiB para el asignador de Torch. Había aplicaciones ajenas concurrentes. No se midieron energía, transferencias físicas ni una mejora predictiva. El estudio de coste emparejado quedó preparado como diseño y no se ejecutó. Tampoco se ejecutó una evaluación financiera nueva ni se abrió el test final.

La validación de paquetes comprueba coherencia de tipos, esquemas, rutas y sumas. Un paquete aislado no acredita la autenticidad de su residual ni de una sustitución coherente de descriptor y lecturas. Su procedencia necesita también el productor y el registro nativo.
