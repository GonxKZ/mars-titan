# Vistas temporales con cobertura macro completa

La edición estricta combina las cuatro modalidades del corpus original con los 140 indicadores del [panel macro compuesto](macro-editions.md). Las vistas conservan los archivos de precios y representaciones. Cada ventana tiene sus etiquetas, exclusiones y manifiesto. El contexto macro se calcula una vez por sesión y sustituye al vector anterior durante la lectura, con máscaras y disponibilidad comprobadas.

La [configuración estricta](../../configs/evaluation/strict-macro-walk-forward.json) se fija a partir de la cobertura disponible. Mantiene separado el protocolo anterior, que proponía tres años de entrenamiento y no cabe en esta edición. La nueva configuración exige al menos seis meses de historia disponible, dos meses de validación, uno de calibración y uno de evaluación. El avance es mensual. La reserva final sigue siendo 2024.

| Ventana | Entrenamiento hasta | Validación | Calibración | Evaluación |
| --- | --- | --- | --- | --- |
| 000 | Mayo de 2023 | Junio y julio | Agosto | Septiembre |
| 001 | Junio de 2023 | Julio y agosto | Septiembre | Octubre |
| 002 | Julio de 2023 | Agosto y septiembre | Octubre | Noviembre |
| 003 | Agosto de 2023 | Septiembre y octubre | Noviembre | Diciembre |

El entrenamiento comienza en noviembre de 2022. Las fechas exactas, la purga por maduración y el margen de una sesión proceden del [contrato temporal](../research/evaluation-protocol.md). Los meses de evaluación no se solapan. Normalización, modelos y demás ajustes aprendidos deben usar el entrenamiento de su ventana. La validación selecciona estados y configuraciones. Calibración y evaluación tienen funciones distintas y no sustituyen esa selección.

## Población comprobada

| Ventana | Entrenamiento | Validación | Calibración | Evaluación |
| --- | ---: | ---: | ---: | ---: |
| 000 | 121.190 | 39.462 | 24.391 | 19.716 |
| 001 | 141.272 | 44.959 | 19.716 | 20.839 |
| 002 | 161.673 | 45.089 | 20.839 | 20.454 |
| 003 | 187.246 | 41.544 | 20.454 | 11.653 |

Estos recuentos corresponden a la edición local `strict140-temporal-20260928-v2`. Se han recorrido sus 940.497 entradas de partición con el lector de entrenamiento. Una muestra puede aparecer en varias ventanas de validación cruzada. Cada evaluación mensual conserva su periodo independiente. La última ventana solo dispone de doce sesiones macro completas, lo que limita la precisión de las comparaciones en diciembre.

La cohorte y la política de evidencia de noticias siguen identificadas por el corpus de origen. La cobertura macro completa no convierte textos candidatos en artículos verificados, no añade conceptos contables ausentes y no valida los ajustes de precios para una simulación financiera histórica. Tampoco demuestra capacidad predictiva. Los resultados de entrenamiento se registran después, con sus propias ejecuciones.

## Preparación, lectura y memoria

`uv run python -m mars_titan.training.temporal_corpus --help` describe la preparación. Recibe manifiesto de origen, protocolo, panel macro, informe de admisión y destino nuevo. Comprueba las huellas, conserva cada exclusión de origen y publica juntas las cuatro vistas. Los objetivos de muestras excluidas quedan nulos. La preparación no abre el test ni inicia modelos.

`CorpusDataset` reconoce el contrato `temporal_view`. Admite las cuatro particiones, comprueba los cortes de las etiquetas y vuelve a verificar el panel macro, la admisión y el origen al comenzar cada recorrido. Las ediciones anteriores conservan su interpretación original de entrenamiento y validación. Los cursores de recuperación siguen ligados a manifiesto, época, semilla y consumo confirmado.

El lector mantiene una caché de buffers de etiquetas validadas y precios. Tiene un presupuesto inicial de un GiB y un máximo de 8.192 entradas. Los buffers son inmutables, se expulsan por uso y solo se reutilizan mientras coincida la identidad del archivo. `cache_bytes=0` permite recorrer la referencia sin caché. El presupuesto limita los buffers retenidos, no representa toda la memoria del proceso.

En dos comparaciones sobre las cuatro particiones de la ventana 000, el recorrido completo pasó de 73,22 y 73,46 segundos sin caché a 47,82 y 45,24 segundos con caché. Incluye construcción del lector, validación de archivos, formación de lotes y comprobación de su contenido. Los hashes de entradas, objetivos, fechas e identificadores coinciden exactamente. La caché retuvo 451,53 MiB y el mayor RSS observado fue 738,33 MiB. Son medidas del suministro de datos, no de un entrenamiento completo de un modelo.

Las pruebas, el perfil, la cobertura y sus límites se registran en [temporal-corpus-quality.json](../../reports/resources/temporal-corpus-quality.json). La ejecución inicial detectó un motivo de exclusión truncado. La edición v2 corrigió ese problema antes de entrenar y conserva el texto completo de los motivos originales.

Dos comparaciones posteriores incluyeron un entrenamiento GRU completo de una época en `cuda:0`, sus evaluaciones y checkpoints. Sin caché tardaron 80,63 y 80,55 segundos. Con caché tardaron 53,95 y 53,53 segundos. Los archivos de predicciones de entrenamiento y validación coinciden exactamente. Estas sondas sirven para comprobar el cambio de rendimiento y no seleccionan hiperparámetros ni sustituyen la campaña científica.
