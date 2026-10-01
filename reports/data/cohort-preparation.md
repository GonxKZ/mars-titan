# Preparación de dos cohortes

La normalización distingue el corpus original auditado y el contraste editorial
externo. Conserva los activos incompletos, los fallos y cada exclusión. Una
preparación terminada no significa que todas las empresas puedan entrenarse.

La [política de cohortes](../../docs/data/cohort-policies.md) define las
diferencias y el tratamiento temporal. El recorrido puede pausarse y reanudarse
por activo. Una estructura contable inválida se excluye antes de construir claves
y no impide preparar las demás empresas.

## Comprobaciones

La batería completa pasó con 1.086 pruebas, sin omisiones, en 117,16 segundos
bajo instrumentación de cobertura. Se detectaron cuatro mutaciones dirigidas
de la admisión editorial. Cobertura, complejidad y CRAP se registran con sus
herramientas y convenciones, sin tratarlos como garantía de ausencia de errores.

La revisión independiente localizó cuatro fallos. Se añadieron pruebas que
reprodujeron la divergencia entre lecturas de una noticia, la alteración del nivel
de evidencia en un recibo recuperado, la pérdida del recorrido por una fuente
desaparecida y una estructura contable inesperada. Las correcciones comprueban
los bytes releídos, la identidad del recibo y la continuación del resto del corpus.

## Memoria y ensayo real

| Registros sintéticos | Bytes de entrada | Tiempo interno | Pico de RAM del proceso |
| --- | ---: | ---: | ---: |
| 5.000 | 11.617.780 | 0,403 s | 157,31 MiB |
| 50.000 | 116.277.780 | 4,518 s | 162,93 MiB |

Se usa un AMD Ryzen 9 8945HS con 32 GB de RAM. El texto sintético es repetitivo
y sirve para comprobar el crecimiento del almacenamiento y la memoria, no para
estimar precisión, compresión del corpus real o rendimiento de GPU. Las medidas
coincidieron con pruebas locales. No se midió energía ni se vació la caché del
sistema operativo.

Un ensayo previo recorrió los cinco primeros candidatos del inventario real.
A y AADR quedaron identificados como incompletos. AA produjo 6.031 filas de
precios, 1.603 noticias y 989 hechos. AAAU produjo 1.353, 7 y 114 respectivamente.
AACG produjo 3.984 precios y 49 noticias, pero ningún hecho admisible. Esa última
empresa no se convierte por ello en una muestra de cuatro modalidades.

El ensayo queda conservado con la identidad del código utilizado. No constituye
una selección de empresas para la campaña ni un entrenamiento. La materialización
multimodal y sus etiquetas requieren sus propias comprobaciones.

La [evidencia de calidad](../resources/cohort-preparation-quality.json) conserva
medidas, recuentos, huellas y límites.
