# Postentrenamiento de las ventanas temporales

Los ajustes posteriores y las referencias tabulares pueden reutilizar las vistas con entrenamiento, validación, calibración y evaluación. Los límites proceden del protocolo verificado de cada ventana, incluida la sesión de purga y la maduración de etiquetas. El corte fijo de enero de 2023 queda reservado al formato histórico de dos particiones.

La preparación ordenada exporta entrenamiento y validación. La rejilla de 21 acciones y la normalización se ajustan con entrenamiento. Los cuatro recuentos declarados se conservan para comprobar que el padre y el ajuste pertenecen a la misma población. La identidad incluye el manifiesto de origen y las implementaciones del calendario y las transformaciones temporales.

El lector original puede decodificar etiquetas de otras particiones al leer un archivo Parquet compartido. Esas filas no se entregan al ajuste. Una prueba cambia los valores de calibración y evaluación y comprueba que la rejilla y los archivos exportados de entrenamiento y validación conservan exactamente sus valores. No se presenta esta comprobación como ausencia de lectura física de columnas compartidas.

La recuperación rechaza cambios en el vínculo con la supervisión y conserva las particiones completas ya confirmadas. Las pruebas también rechazan cohortes dentro de la purga, poblaciones incompatibles y padres cuya implementación temporal no corresponde a sus huellas.

La serie histórica y las ventanas con 140 indicadores son ediciones distintas. No se mezclan sus poblaciones, métricas ni checkpoints. La continuación real de edición 2 utiliza el diseño de [selección con estado inicial y paciencia](continuation-selection.md). No modifica el protocolo de una campaña histórica en curso.
