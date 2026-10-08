# Protocolo pendiente de integración de CM-v1

Las funciones matemáticas tienen pruebas técnicas, pero no constituyen todavía cuatro variantes ejecutables de un sistema. Antes de comparar B, B+C, B+M y B+C+M hay que fijar B mediante arquitectura, commit, configuración, datos, semillas y estado inicial. Titans-MAC será el núcleo nuevo. La GRU episódica se conserva como referencia con identidad propia.

C utiliza `RᵀJR`, donde J deriva la transición conjunta de pesos rápidos y momentum de MAC. La [definición de F](mac_local_control.md) incluye lectura previa, atención y actualización asociativa. La medida comprimida puede omitir direcciones expansivas y su penalización sigue siendo heurística. Diagnóstico y penalización identifican rango, rejilla, umbral, peso, frecuencia, selección lógica y precisión. B del nuevo factorial debe usar el mismo backend Math de MAC y la misma base, con modo disabled. No se cambia la identidad del predictor anterior con `local_control=None`.

M necesita un snapshot del banco con representación congelada y versionada, candidatos autorizados y etiquetas maduras. Esa elegibilidad debe comprobarse antes de llamar a la API. Una consolidación no puede hacer visible una etiqueta pendiente. El representante conserva el episodio real y su procedencia. El banco episódico, la memoria neural de Titans-MAC y los parámetros persistentes son estados diferentes.

La integración debe registrar capacidad, métrica, política de candidatos, desempates, límites de trabajo, sustituciones y coste de cobertura sobre el mismo E y F. Debe poder recuperar selección, estados de memoria, etiquetas pendientes, RNG y cursor confirmado. La serialización y el contrato temporal no forman parte de estas dos funciones.

La comparación propuesta utiliza las mismas filas, máscaras, semillas y presupuesto de actualizaciones. El contraste principal emplea K=1. C y M desactivados deben reproducir B, incluidos estado, RNG y recuperación. Las pruebas de C ya cubren el predictor financiero y su estado rápido. La paridad del factorial que incluya el ciclo episódico completo permanece pendiente.

Antes de cualquier estudio se deben completar pruebas temporales, recuperación y controles CPU/CUDA de los caminos utilizados. El protocolo de selección debe permanecer temporal, con el test final cerrado y reglas de parada y retención declaradas. La selección del mejor estado debe admitir el padre inicial cuando los ajustes posteriores empeoren. No se ha ejecutado aprendizaje ni se ha estimado una mejora predictiva para CM-v1.
