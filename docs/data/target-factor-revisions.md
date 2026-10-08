# Revisiones del factor de las etiquetas

La ampliación de un factor, como CSI 300, puede cambiar el ajuste residual sin cambiar las entradas observadas. `prepare_corpus_targets` admite el argumento opcional `target_factors: Path | None`. Con `None` conserva el recorrido anterior. Con un descriptor explícito crea otra identidad de supervisión y mantiene el manifiesto codificado original.

El descriptor JSON debe identificar todos los mercados del censo. Cada factor conserva instrumento, unidad y metadatos económicos. Solo pueden variar `prices_path` y `prices_sha256`. La revisión exige un corpus completo confirmado y concilia cobertura, recibos y número real de filas, incluidos los candidatos codificados con cero muestras. No permite usar esta opción para declarar completo un prefijo parcial.

`configuration.target_factor_revision` fija el manifiesto padre, el descriptor y la huella del contrato. Las fuentes anteriores y nuevas se verifican antes de preparar y confirmar. El lector vuelve a comprobar la identidad al abrir y al comenzar un recorrido. Las vistas conservan esta dependencia. La revisión no modifica los embeddings ni sustituye las configuraciones de ediciones anteriores.

Las salidas deben estar separadas de las fuentes. Los enlaces en destinos y sus directorios intermedios se rechazan antes de escribir. Un recibo existente vacío, cambiado o incompatible impide recuperar. Las huellas fijadas no se reemplazan por las de un archivo modificado durante la preparación. Estas comprobaciones detectan cambios observados en las fronteras declaradas, sin aportar un bloqueo general contra modificaciones concurrentes posteriores del sistema de archivos.

La API está en `mars_titan.training.corpus_targets.prepare_corpus_targets` y se mantiene accesible desde `corpus_inputs`. Recibe manifiesto codificado, origen preparado y destino nuevo. La selección del backend y la política de entradas conservan sus argumentos existentes. La comprobación reproducible es `uv run pytest tests/training/test_target_factor_revision.py`.

La [verificación del 8 de octubre](../../reports/data/target-factor-revision-verification-20261008.json) ejecutó 298 pruebas CPU relacionadas. La revisión independiente ejecutó 44 pruebas focales y seis reproducciones y confirmó la corrección de cinco defectos. La cobertura y la complejidad se publican como diagnósticos. El mayor CCN modificado sigue siendo 98 y no se presenta como código de baja complejidad.

No se han generado objetivos reales con esta revisión ni se ha medido su coste sobre el corpus completo. La edición histórica sigue en codificación y el bloqueo de aprendizaje permanece activo. La ampliación retrospectiva de CSI 300 conserva sus límites de disponibilidad y de simulación financiera.
