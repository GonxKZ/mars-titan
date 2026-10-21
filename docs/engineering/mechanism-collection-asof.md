# Leer evidencias de mecanismos por corte de recepción

`FinancialSession.mechanism_collection_at` recupera una colección de alternativas y reconstruye sus resultados desde registros nativos confirmados. Sirve para preparar ventanas temporales dentro de una sesión más larga. No ajusta parámetros ni sustituye la selección y evaluación de una política.

El llamante indica la generación de la decisión, `through_generation` inclusiva y `before` exclusivo. Todas las generaciones elegidas deben estar confirmadas y tener un corte anterior a `before`. Una generación que alcanza ese límite provoca un error. El lector no busca el límite inspeccionando etiquetas posteriores.

La vista vuelve a leer el paquete previo a etiquetas, comprueba su ancla contra la emisión nativa y recorre el prefijo elegido. Verifica la huella y el contenido de cada registro contra la traza confirmada. Reconstruye las etiquetas, exclusiones y censuras de esa decisión por identidad de flujo y fecha. No usa el recibo actual de la colección ni abre registros nativos posteriores a `through_generation`.

Una etiqueta puede tener disponibilidad declarada el martes y recibirse el viernes. Una vista que termine el miércoles la conserva como pendiente. El reloj de `supervision` sigue siendo el máximo de las disponibilidades recibidas, mientras que `as_of.native_records` conserva los cortes reales de recepción. Las dos fechas tienen funciones distintas. Una exclusión conserva `value=None` y su evidencia. Una censura conserva el cierre administrativo y nunca se transforma en etiqueta ni en error cero.

La salida contiene `collection`, `supervision`, `censored` y `as_of`. Este último incluye generaciones, cortes, huellas y bytes de los registros, correspondencia entre filas y recepciones y una huella de la vista. La huella incorpora la identidad del paquete y el contenido reconstruido. No depende del recibo ni de la cabeza posterior del archivo. Es una comprobación de integridad, no una firma de un tercero ni una acreditación de aprendizaje.

El método comprueba al terminar que la cabeza confirmada no ha cambiado durante la lectura. No escribe artefactos, publica estados ni actualiza generadores aleatorios. La recuperación y la retirada de checkpoints antiguos no eliminan las referencias directas de las colecciones ni sus registros nativos.

## Límites y coste

Por defecto admite hasta 256 registros y 64 MiB de paquete y registros nativos. El máximo configurable es 256 registros y 1 GiB. El presupuesto restante limita cada lectura antes de abrir el registro, además del límite nativo de 64 MiB por archivo. Agotar un límite provoca un error, sin devolver un prefijo truncado.

Los bloques de trazas tienen sus propios límites del archivo y no se incluyen en `source_bytes`. Su validación puede cargar un bloque que también contiene trazas posteriores. Esos bytes se usan para comprobar integridad. No se toman resultados posteriores para construir la supervisión. Por tanto, `source_bytes` no representa todos los bytes físicos leídos ni el pico de memoria del proceso.

La reconstrucción individual recorre registros de forma secuencial y conserva el paquete, un registro actual y resultados de sus filas. Pedir muchas vistas por separado repite parte de ese trabajo.

## Lectura conjunta

`FinancialSession.mechanism_collections_at(generations, through_generation=..., before=...)` reconstruye varias colecciones con un único recorrido de registros nativos. `generations` es una tupla explícita, ordenada y sin repeticiones, con entre una y 256 generaciones de decisión. No tiene que ser consecutiva. Las generaciones de liquidación no contienen una colección y se rechazan si se solicitan como tal.

Cada registro nativo entre la primera generación elegida y `through_generation` se lee una vez. Un índice de flujo y fecha dirige cada resultado a la colección que lo emitió. El lector comprueba por separado el ancla de cada paquete. Conserva una sola lista de procedencia de registros, en lugar de repetirla para cada colección. Los recibos actuales y los paquetes no seleccionados no se leen. Los bloques del historial siguen sujetos a la comprobación de integridad descrita antes.

La salida contiene `collections` y `as_of`. Cada elemento de `collections` incluye `generation`, `collection`, `supervision`, `censored`, `resolutions` y `packet_reference`. `as_of` identifica el contrato, el modelo, las generaciones, el corte exclusivo, los registros y la huella `window_sha256`. La huella enlaza los paquetes y los resultados reconstruidos con ese prefijo. No acredita que un selector se haya ajustado, ni autentica a un equipo capaz de reescribir código y artefactos.

Los límites de registros y bytes se aplican al conjunto, con los mismos valores por defecto y máximos que la vista individual. `source_bytes` cuenta cada paquete seleccionado y cada registro nativo una sola vez. No incluye los bloques de trazas ni equivale a bytes físicos leídos del almacenamiento. Agotar un límite rechaza toda la vista. No se devuelve una selección truncada.

El lector conserva todos los paquetes seleccionados y sus resultados porque debe devolverlos juntos. Esto puede aumentar el pico de memoria frente a consumir y liberar vistas individuales de una en una. Mantiene solo un registro nativo completo a la vez. La lista de procedencia común contiene metadatos y no copia su contenido. La comprobación de cabeza se realiza después del recorrido completo, antes de devolver la ventana. Las dos APIs usan el mismo motor de recepción y la vista individual conserva su esquema.

## Comprobación y uso posterior

Las pruebas técnicas están en `tests/memory/test_mechanism_collection_asof.py` y `tests/memory/test_mechanism_collection_window.py`. Cubren recepciones tardías, límite exclusivo, disponibilidades desordenadas, identidad por fecha, pérdidas sobre las mismas filas maduras, exclusiones, censuras, agotamiento, integridad y recuperación en CPU y CUDA. La ventana añade selección no consecutiva, igualdad con vistas individuales, lectura única de cada registro y procedencia común. Los datos y resultados financieros empleados son fixtures identificados. No hay ajuste del predictor ni de un selector.

El [informe de lectura histórica](../../reports/engineering/mechanism-asof-20261021/README.md) recoge las ejecuciones y sus límites. El [ajuste preparado](mechanism-policy-fitting.md) sigue exigiendo sesiones completas cerradas dentro del tramo. Incorporar esta vista a aquel lector exige un contrato de fuentes y ventanas propio, además de probar su recorrido positivo cuando se permita. La nueva API no acredita que ese ajuste se haya ejecutado ni que mejore las predicciones.
