# Ruido, incertidumbre y fiabilidad de las predicciones

La salida principal de MARS-TITAN es numérica. No se plantea generar explicaciones económicas libres y tratarlas como hechos. Una predicción puede estar equivocada aunque no contenga texto inventado. La fiabilidad se evalúa por errores, calibración, abstención y procedencia, junto con la corrección del programa.

## Qué información puede corregirse

Se distinguen errores de datos, componentes predecibles de variación y cambios reales del mercado. Un precio incoherente por una acción corporativa mal tratada requiere corregir su procedencia. Un salto válido tras una noticia no se elimina simplemente por ser extremo. Recortar todos los extremos podría suprimir precisamente los eventos que la memoria pretende estudiar.

Las transformaciones candidatas incluyen normalización robusta ajustada en entrenamiento, pérdidas resistentes a errores grandes y filtrado estrictamente causal. Cada una tendrá una comparación con datos sin ese tratamiento. Se medirá el retraso que introduce el filtro y el efecto sobre colas, eventos y calibración. Se excluyen suavizadores centrados, interpolación que utiliza observaciones posteriores y estados de régimen obtenidos con toda la serie cuando se usan como entradas históricas.

El [contrato temporal](../data/data-contract.md) exige que los cambios futuros no alteren las entradas anteriores. La [revisión adversarial](adversarial-review.md) incluye contraejemplos para filtros, revisiones y memoria. No se promete eliminar toda la incertidumbre futura mediante limpieza.

## Controles antes de emitir una salida

La predicción debe corresponder a un activo, horizonte, moneda, versión de modelo y corte identificados. Los valores deben ser finitos y los cuantiles respetar su orden. Las ausencias se comunican mediante máscaras y motivos. No se sustituyen silenciosamente por cero, por una noticia generada o por el último valor de otro activo.

El sistema deberá distinguir una salida válida, una salida de baja confianza, un estado demasiado antiguo y una entrada inválida. La respuesta ante datos incompatibles o un presupuesto agotado será explícita. La abstención se contabiliza en la cobertura y la exposición de la simulación, para no ocultar fallos retirando los casos difíciles del denominador.

La calibración se examina por ventana y, cuando haya muestra suficiente, por activo y régimen. Un promedio bien calibrado puede ocultar errores graves en un periodo. La incertidumbre de un modelo no acredita seguridad económica ni representa una probabilidad de beneficio garantizada.

## Datos externos y manipulación

Los documentos y noticias recibidos son datos. Su texto no autoriza ejecutar instrucciones, cambiar configuraciones o llamar servicios. Las cifras fundamentales se obtienen preferentemente de estructuras de la fuente, con unidades y fechas contrastadas. Una extracción automática necesita validación contra el documento, no solo una respuesta plausible de un modelo de lenguaje.

Se probarán duplicación masiva de una noticia, fechas alteradas, unidades erróneas, series revisadas, ausencia de una modalidad y valores fuera de dominio. Las pruebas conservarán un conjunto limpio de comparación y no modificarán el benchmark original. El protocolo medirá cuánto se degrada el error, cuántos casos se detectan y cuántas observaciones válidas se rechazan por error.

Las fuentes gratuitas complementarias se guardarán en instantáneas separadas. Una descarga reciente no demuestra que ese dato fuera conocido en una fecha histórica. El acceso público tampoco sustituye la comprobación de derechos ni garantiza continuidad del servicio.

## Qué permitirá afirmar una mejora de fiabilidad

Se exigirá una disminución de error o de fallos concretos, con cobertura, costes e incertidumbre reportados y sin degradar silenciosamente otro objetivo. La capacidad de recuperar un entrenamiento, rechazar una entrada inválida o reconstruir una predicción es verificable. La frase «funciona perfectamente» no sustituye esos criterios.
