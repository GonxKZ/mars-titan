# Procedencia de los codificadores congelados

## Configuración usada

La extracción mantiene MiniLM multilingüe para texto y ResNet18 para imágenes.
Son representaciones congeladas para medir y comparar modelos compactos. No se
entrenan estos codificadores ni se interpreta su uso como reproducción histórica
estricta del conocimiento disponible en cada fecha del mercado.

| Elemento | Decisión y evidencia |
| --- | --- |
| Texto | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revisión `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, 384 dimensiones. |
| Tokenización | Se fijan `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` y `config.json` mediante SHA-256. También se identifica el tokenizador cargado en memoria. |
| Texto largo | Fragmentos de hasta 126 tokens de contenido, con tokens especiales explícitos. Se procesa el documento completo y se agregan los fragmentos ponderando por longitud. |
| Imagen | `resnet18.IMAGENET1K_V1`, pesos identificados por URL y SHA-256, 512 dimensiones después de retirar el clasificador. |
| Transformación visual | Gráfico regenerado de 224 × 224 píxeles, RGB y normalización fijada. No se recortan los extremos de la ventana. |
| Ejecución | `cuda:0`, FP32, modo evaluación y gradientes desactivados. Sin alternativa silenciosa en CPU. |

La [documentación de Sentence Transformers](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
describe el modelo multilingüe y su entrenamiento con datos paralelos. No ofrece
en esa página una auditoría de todos sus documentos ni una fecha máxima verificable
de conocimiento. No se presume ausencia de solapamiento con textos financieros.

Los [metadatos de la revisión fijada](https://huggingface.co/api/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/revision/e8f8c211226b894fcb81acc59f3b34ba3efd5f42)
declaran licencia Apache 2.0, creación del repositorio el 2 de marzo de 2022 y
última modificación el 28 de enero de 2026. Esas fechas describen el repositorio,
no certifican la fecha de entrenamiento ni que estos mismos pesos estuvieran
disponibles antes. Se mantienen `historical_simulation = false` y esa limitación
en los informes.

La [ficha oficial de ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
identifica los pesos `IMAGENET1K_V1`. El rendimiento publicado en clasificación de
imágenes no demuestra utilidad sobre gráficos bursátiles. El uso y los derechos
de los datos originales de preentrenamiento son distintos de la licencia del
código de MARS-TITAN.

## Identidad de caché

La huella del codificador incluye pesos, revisión, artefactos de tokenización,
configuración, transformación, precisión, dispositivo, código y versiones de
ejecución. La clave de texto añade contenido y regla de disponibilidad. La imagen
se identifica por los bytes del gráfico pasado que realmente se codifica.

Estas huellas se incorporan a las claves nuevas. Cambiar un artefacto invalida la
reutilización de su representación. No se sobrescriben ni se certifican
retroactivamente las cachés antiguas de las sondas de coste. Esos ensayos siguen
asociados a su configuración y revisión originales.

Los ejemplos almacenados conservan además la decisión temporal y la procedencia
de las modalidades. Compartir una representación idéntica no autoriza compartir
una muestra si cambia su disponibilidad o deja de ser admisible.

## Control sin preentrenamiento posterior

La sensibilidad propuesta conserva las cuatro modalidades y la misma población
admitida. Para texto se propone un hashing determinista de n-gramas, sin vocabulario
aprendido con todo el periodo. Para imagen, píxeles reducidos mediante una regla
fija o una red pequeña inicializada desde cero y entrenada solo dentro de cada
ventana pasada. Precios, fundamentales y macro no cambian.

La dimensión, normalización y presupuesto se fijarán con entrenamiento y validación,
nunca con el test. Se comparará esa alternativa con los codificadores congelados
en las mismas particiones. No se atribuirá toda diferencia al preentrenamiento,
porque también cambian capacidad y representación. Esa sensibilidad todavía no
se ha ejecutado y no permite cerrar #10 como comparación terminada.

## Límite de la preparación

Comprobar huellas y repetibilidad no verifica el contenido de las noticias. La
incidencia de correspondencia entre título y cuerpo descrita en la [política de
noticias](news-policy.md) sigue condicionando la extracción definitiva. No se
vuelve a codificar el corpus ni se inicia entrenamiento confirmatorio con esta
incorporación.

## Comprobación ejecutada

La suite completa pasó con 364 pruebas, incluida una extracción real en CUDA.
Se repitieron dos textos sintéticos, uno de ellos de varios fragmentos, y un
gráfico sintético. En los tres casos la diferencia máxima fue cero, dentro de
las tolerancias fijadas de `rtol = atol = 10⁻⁶`. Las huellas de todos los pesos y
buffers fueron iguales antes y después. No se usaron etiquetas financieras.

El [recibo de la ejecución aislada](../../reports/data/encoder-provenance.json)
registra hashes y versiones. En ese proceso, la inicialización tardó 3,87 segundos
y la repetición conjunta de los dos textos y el gráfico, 17,81 milisegundos.
Solo hay una observación temporizada. No es una estimación de percentiles ni del
coste por artículo del corpus. El pico fue de 546.401.792 bytes asignados y
578.813.952 reservados en CUDA, con 2.584,48 MiB de RSS del proceso, incluida la
comprobación de los estados completos.

Las pruebas de caché comprueban los cuatro artefactos por separado. Una mutación
que sustituye su SHA-256 por un valor constante se detectó en los cuatro casos.
Las diez pruebas dirigidas, incluida CUDA, cubrieron 114 de 121 sentencias y
23 de 30 ramas del módulo, un 90,73 % combinado con coverage.py 7.16.1.
La función nueva de huellas tiene complejidad 2, cobertura de sentencias del
100 % y CRAP 2, según Radon 6.0.1. Son controles acotados, no una certificación
de que cualquier entrada producirá una representación útil.

```bash
MARS_ENCODER_INTEGRATION=1 HF_HUB_OFFLINE=1 uv run --locked \
  --extra cuda --extra encoders pytest \
  tests/data/test_encoder_provenance_integration.py -q -s
```

La comprobación requiere los pesos ya descargados. Sin la variable de activación
queda omitida de la suite habitual. Al activarla, la falta de CUDA produce un
error, no una ejecución alternativa en CPU.
