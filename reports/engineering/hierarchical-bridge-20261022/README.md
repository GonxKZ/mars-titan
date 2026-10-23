# Jerarquía episódica de cuatro vistas, dos grupos y una raíz

La variante opcional de [#719](https://github.com/GonxKZ/mars-titan/issues/719) conserva un snapshot canónico, una selección común y una proyección de valores por lectura. Reutiliza las cuatro matrices regionales. Los grupos y la raíz son estados de trabajo que se reinician en cada lectura, sin parámetros aprendibles propios. Todos los controles de este contraste tienen cuatro vistas y su propia identidad. Las referencias históricas se conservan. La [especificación](../../../docs/experiments/hierarchical-evidence-bridge-v1.md) recoge ecuaciones, fuentes y diferencias entre árbol, estrella y control sin realimentación. La opción permanece desactivada por defecto.

Las comprobaciones son técnicas y usan fixtures identificados. No se ha entrenado ni evaluado un predictor sobre datos financieros en esta ronda. No hay una mejora predictiva medida ni una selección frente a SOTA. El bloqueo de aprendizaje y el test final permanecen intactos. La integración del trabajo publicado en ramas sigue pendiente de revisión.

## Código y comprobaciones

`bridge_hierarchy.py` ejecuta dos rondas grupo → raíz → grupo → pasos locales. Las vistas aplican sus matrices y tanh. Los auxiliares utilizan medias e identidad, sin tanh. L1, L2 y L4 cuentan 18, 26 y 42 actualizaciones nuevas por lectura. Las cuatro vistas iniciales se cuentan aparte. La reserva conjunta de colección y emisión admite hasta 38 lecturas, sin renovar cuotas por ruta o bloque. Se comprobaron los bordes 988/987, 1596/1595 y K4 directo 104/103 antes de MAC.

La fábrica, el catálogo, los consumidores congelados, los checkpoints y las dos trazas identifican la jerarquía. El resultado conserva los estados auxiliares realmente calculados y el grafo efectivo. El bloque regional 4×4 es una submatriz del operador 7×7, sin renormalización. Los checkpoints incompatibles se rechazan antes de copiar parámetros del padre. Los estados auxiliares no se incorporan a los pesos rápidos ni a otro banco.

[verification.json](verification.json) reúne 685 IDs de prueba únicos, sin fallos, errores ni omisiones. Son 676 casos de la selección principal, ocho guardas adicionales y una prueba adicional de integridad de artefactos. Los tres casos del CLI repetidos con esta última no se suman otra vez. Se ejercitaron 207 parametrizaciones `cuda:0` y tres comparaciones explícitas CPU/CUDA. El recuento CUDA es 210. La selección incluye recuperación nativa y siguiente emisión, snapshots, máscaras, banco vacío y ausente, orden de activos, bloques y RNG. No es la suite completa del repositorio.

La referencia apagada se capturó antes de editar. Conserva exactamente identidad de configuración, claves y valores del checkpoint, RNG y salidas CPU/CUDA al omitir la opción o declarar None. Los sellos de fuentes y los identificadores de runtime sí cambian con el código. La recuperación de sesiones históricas necesita su runtime conservado.

[quality.json](quality.json) usa coverage.py 7.16.1 y Radon 6.0.1. En los once módulos seleccionados cubre 1.737 de 2.027 sentencias y 707 de 926 ramas. El módulo nuevo cubre 92/92 sentencias y 38/38 ramas. CRAP se calcula como `C² × (1 − cobertura de sentencias)³ + C`. El máximo del núcleo nuevo es 27. La función antigua `_associative`, fuera de los recorridos de esta selección, conserva CRAP 132 y cobertura nula en esta medida. No se oculta ni se interpreta la cobertura como ausencia de defectos.

[mutations.json](mutations.json) registra diez mutaciones dirigidas detectadas, con controles inicial y restaurado de 153/153 casos. Incluyen identidad, calendario, reinicio entre rondas, normalización sin realimentación, gradientes auxiliares, cuota global, grafo archivado, grupos exportados, carga y doble envío del settlement. Los fallos fueron de la fase de ejecución prevista, sin errores de preparación, omisiones o timeouts. Se comprobaron procedencia y bytes importados. La copia quedó restaurada y el worktree intacto. Es el resultado de esas diez sustituciones, no una campaña de mutación de todo el repositorio. El último mutante actúa en la llamada al ejecutor nativo y no certifica su registro interno de idempotencia C++.

Los verificadores de trazas comprueban estructura, finitud, formas, grafos y ceros de ausencia. No recomputan ni autentican cualquier valor finito de un auxiliar poblado. La integridad persistida depende de las referencias confirmadas: `SessionArtifacts` contrasta tamaño y SHA antes de cargar o devolver una copia de caché. La prueba adicional altera un valor finito y restaura el mtime, y comprueba que la referencia original siga siendo rechazada.

## Casos analíticos

[analytical.json](analytical.json) conserva tres paneles FP64 ejecutados en CUDA, con entradas, matrices, operadores y estados. Una señal atraviesa los dos grupos por la raíz y deja de hacerlo al cortar su camino. Un control directo recupera el signo sin jerarquía, por lo que ese fixture no demuestra que el árbol sea necesario.

En el caso de cancelación, el error cuadrado de reconstrucción del árbol es 0,8192438615 y el del control local fijado es cero. Son objetivos construidos para comprobar el mecanismo, no errores financieros. El Jacobiano en otro fixture tiene sensibilidad `3521/576 = 6,1128472222`, aunque sus estados sean cero. El control sin realimentación tiene referencia 16 en ese ejemplo. Ninguno de esos números acredita estabilidad global o una ventaja predictiva.

## Coste observado

[cost.json](cost.json) contiene 96 registros: 24 condiciones por etapa, en dos órdenes. Hay 624 invocaciones medidas o de calentamiento del puente FP32 y 624 cohortes de preparación FP64. Estas últimas ejecutan 936 llamadas físicas a `FrozenFinancialConsumer.prepare`. Las cohortes de 256 filas se dividen en dos bloques de 128 en todas las variantes. Se conservan todas las filas y el snapshot común. La referencia CPU FP64 también utiliza bloques de 128, fuera del tiempo medido.

La ejecución usa una RTX 4070 Laptop de 8.188 MiB, PyTorch 2.14.0+cu130, CUDA 13.0 y driver 595.91.07. El equipo tiene un Ryzen 9 8945HS, 16 CPU lógicas y 32.313.376.768 bytes de RAM física. Los parámetros están emparejados dentro de cada etapa. Las etapas FP32 y FP64 no se presentan como una misma instancia convertida. Se mantienen TF32 y autocast desactivados. Cada condición tiene tres calentamientos y diez observaciones. Los percentiles son descriptivos y no estimaciones estables de cola.

Para 256 filas y 1.024 episodios, las medianas en ms fueron:

| Variante | Puente, orden inicial / inverso | Preparación por dos bloques, orden inicial / inverso |
| --- | ---: | ---: |
| Plano J4 | 2,599 / 2,627 | 23,603 / 19,700 |
| Árbol L1 | 3,714 / 3,689 | 25,878 / 20,667 |
| Árbol L2 | 4,334 / 4,355 | 25,929 / 21,093 |
| Árbol L4 | 5,250 / 5,949 | 28,181 / 23,336 |
| Estrella L2 | 4,272 / 4,223 | 25,112 / 22,510 |
| Sin realimentación L2 | 4,324 / 4,383 | 20,927 / 21,234 |

El árbol añade tiempo respecto al plano en esta carga. La variación entre órdenes impide atribuir pequeñas diferencias entre variantes a una aceleración estable. El control sin realimentación calcula los auxiliares, pero cambia la función efectiva y su recorrido de gradientes. Igualar parámetros no iguala cómputo o capacidad.

El puente FP32 tiene 82.432 bytes de parámetros en todas las variantes. Los buffers pasan de 16 a 65 bytes. En la carga anterior, el máximo adicional observado por el asignador durante el puente fue 6.457.344 bytes para J4 y 7.507.968 para árbol L2. Durante la cohorte de preparación fue 18.415.616 y 18.547.712 bytes respectivamente. Estas cifras incluyen calentamiento e inicialización de bibliotecas. No son memoria exclusiva de cada modelo y no se interpretan como ahorro. El pico de proceso observado fue 109.154.304 bytes asignados en GPU, 123.731.968 reservados y 1.649.242.112 de RSS, incluyendo modelos residentes, referencias, instrumentación y runtime.

La preparación medida incluye verificación ligera, codificadores del fixture, MAC, lector, cabeza y estados propuestos. Retiene los resultados de los bloques como tupla. No incluye construcción de entradas, codificación del corpus, unión posterior de resultados, colección, persistencia nativa, feedback ni exportación de pruebas. No se midieron entrenamiento, backward del predictor, energía o tráfico físico. Los bytes lógicos y los contadores de nodos no son FLOPs ni transferencias medidas.

[profiles.json](profiles.json) separa categorías del profiler. Una cohorte instrumentada registra 750 eventos de kernel para J4 y 946 para árbol L2. Sus duraciones no se suman con las atribuciones de los operadores CPU ni se usan como latencia sin instrumentación. Las trazas completas quedan fuera de Git. No se añadieron kernels propios ni se sustituyeron las primitivas de PyTorch a partir de ese único perfil.

## Trazabilidad y límites

El primer intento de coste se detuvo en la referencia CPU FP64 por la guarda de trabajo de 32 MiB. El segundo llegó a la carga mayor y se detuvo por la misma guarda del lector completo FP64. Ambos están conservados como incompletos. Antes del tercer intento se fijaron los bloques de 128 para las comprobaciones CPU y para todas las preparaciones. No se elevaron los límites ni se descartaron filas. La versión final completó los 96 registros en 25,696 s internos y 26,579 s de proceso supervisado.

También se conservan el RED de configuración, los cuatro fallos esperados de integración y sus dos errores previos de importación del fixture. El primer Ruff sin seleccionar el entorno no encontró su ejecutable. El formato se corrigió antes de las comprobaciones finales. Un exportador de coverage intentó leer como base de datos un JSON con el mismo prefijo. La captura definitiva parte de una copia de la base de datos sin esa colisión y conserva los mismos recuentos ampliados.

Los comandos supervisados y sus retornos constan en `verification.json`, con rutas representadas mediante variables de entorno. Los recibos originales sin editar quedan fuera de Git, identificados por SHA. [hierarchical_bridge_cuda.py](../../../benchmarks/hierarchical_bridge_cuda.py) imprime el protocolo sin importar Torch cuando se llama sin `--run`. La ejecución comprobada utiliza una copia privada, `--repository` y una salida nueva, con supervisor de 180 s por proceso y cuota técnica global de 900 s. Las huellas y tamaños de los JUnit, logs, referencias, guiones, trazas y controles están en [external-artifacts.json](external-artifacts.json). Los artefactos voluminosos permanecen fuera del repositorio.
