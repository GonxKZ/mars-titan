# Libros para sostener las decisiones del proyecto

La biblioteca combina fundamentos y consulta aplicada. Descargar un libro no equivale a haberlo leído íntegramente: las siguientes entradas identifican el uso previsto y los capítulos que conviene trabajar, sin atribuir resultados al proyecto.

| Libro y fuente del autor | Lectura dirigida | Decisión del proyecto |
| --- | --- | --- |
| James et al., [An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/) | Regresión, regularización, remuestreo, árboles y comparación múltiple. | Construir referencias sencillas y justificar la selección. La validación financiera debe conservar el tiempo aunque un ejemplo general use particiones aleatorias. |
| Hastie et al., [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/main.html) | Sesgo-varianza, selección de modelos, boosting y métodos de conjunto. | Explicar por qué una arquitectura compleja necesita controles de capacidad y de búsqueda. |
| Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/) | Funciones de pérdida, entrenamiento, regularización, atención y evaluación. | Fundamentar la arquitectura compacta y distinguir dificultades de optimización de límites de generalización. |
| Hernán y Robins, [Causal Inference: What If](https://miguelhernan.org/whatifbook) | Primera parte: preguntas causales, supuestos e identificación. Después, confusión dependiente del tiempo. | Delimitar el significado de «causal»: una red que respeta el pasado no identifica automáticamente efectos de intervenciones. |
| Zhang et al., [Dive into Deep Learning](https://arxiv.org/abs/2106.11342v5) | Tensores, diferenciación, secuencias, atención y rendimiento computacional. | Preparar implementación y medición posteriores. Las dependencias se gestionarán con uv. |
| Hyndman y Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/) | Exploración, referencias ingenuas, residuos y evaluación temporal. | Comprobar primero referencias sencillas y estudiar el error por ventana temporal. Sus ejemplos en R no obligan a añadir otro lenguaje al repositorio. |

Los libros financieros comerciales de López de Prado se registran en la [revisión financiera](finance-review.md), con acceso editorial o por biblioteca. No se incluyen copias obtenidas de sitios sin procedencia acreditada.

## Registro de lectura

Al estudiar un capítulo, anotar edición, páginas, idea utilizada, limitación y decisión afectada en la ficha correspondiente de `reports/`. No convertir esta selección inicial en una lista de lecturas supuestamente terminadas. Los catálogos separan fecha bibliográfica de versión descargada. El manifiesto identifica el archivo concreto mediante SHA-256.
