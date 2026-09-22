# Seguimiento de las referencias de X

La [revisión ampliada](social-followup.md) completa esta primera inspección con la lista del propio autor, las imágenes y las condiciones actuales de cada proveedor. Incluye además dos publicaciones recibidas posteriormente.

Autor: Gonzalo. Verificación: 18 de septiembre de 2026.

Los tres enlaces permiten localizar herramientas, pero tienen distinto valor para el proyecto. Esta nota separa el contenido del post, los recursos originales identificados y lo que todavía no puede darse por comprobado. Las afirmaciones técnicas se contrastan con los proyectos. Una publicación social no se trata como evidencia de rentabilidad.

La lectura directa de X devolvió HTTP 403. El servicio público oficial `publish.twitter.com/oembed`, consultado siguiendo su redirección y sin autenticación, sí devolvió autor, URL, fecha y texto de los tres posts. En los dos posts largos, el texto termina truncado. Una búsqueda inicial por identificador exacto no devolvió resultados. Las búsquedas posteriores por contenido localizaron recursos adicionales. No se ha completado el texto ausente por inferencia.

## Quant Science: librerías de datos

Referencia: [post de @quantscience_](https://x.com/quantscience_/status/2100252394944626797), 16 de septiembre de 2026. [Consulta oficial oEmbed](https://publish.twitter.com/oembed?url=https%3A%2F%2Ftwitter.com%2Fquantscience_%2Fstatus%2F2100252394944626797).

El texto inicial anuncia doce librerías de Python para acceder a datos de mercado. El [hilo reconstruido en Rattibha](https://en.rattibha.com/thread/2100252394944626797), utilizado como pista secundaria, enumera: yfinance, pandas-datareader, IBApi, Alpha Vantage, Nasdaq Data Link, Twelve Data, Polygon, Tradier, Alpaca-py, Finnhub, marketstack y Tiingo. Esa reconstrucción no verifica por sí misma los límites, precios o licencias de los proveedores.

Contrastes concretos con fuentes originales:

| Recurso | Evidencia primaria consultada | Implicación para el proyecto |
| --- | --- | --- |
| [yfinance](https://github.com/ranaroussi/yfinance) | El proyecto distingue su licencia Apache de los derechos sobre los datos de Yahoo y advierte del uso personal de su API | Candidato de exploración. No acredita derechos de redistribución ni un historial point-in-time |
| [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html) | La documentación actual incluye FRED y la biblioteca Fama–French entre sus fuentes mantenidas | Posible acceso a factores y contexto macro, conservando versiones y fechas de publicación |
| [Alpha Vantage](https://www.alphavantage.co/support/) | El proveedor describe acceso mediante clave y planes con distintas condiciones | La disponibilidad gratuita debe revisarse por endpoint, frecuencia y cobertura |
| [Massive, cliente oficial Python](https://github.com/massive-com/client-python) | El enlace del hilo a Polygon conduce al proyecto actual de Massive | Registrar proveedor y versión actuales, sin asumir que el nombre y condiciones del hilo siguen iguales |

La aportación es localizar candidatos para la fase 1. Antes de elegir uno hacen falta cobertura histórica de empresas desaparecidas, acciones corporativas, timestamps, revisiones, condiciones de uso y coste de la extracción necesaria. Esta comprobación no selecciona un proveedor ni ha ejecutado sus librerías. No se han contrastado de forma exhaustiva los doce servicios.

## QuantIndicator: proyectos de LuxAlgo

Referencia: [post de @QuantIndicator](https://x.com/QuantIndicator/status/2099518866799898755), 14 de septiembre de 2026. [Consulta oficial oEmbed](https://publish.twitter.com/oembed?url=https%3A%2F%2Ftwitter.com%2FQuantIndicator%2Fstatus%2F2099518866799898755).

El fragmento recuperado anuncia herramientas públicas de LuxAlgo para gráficos, Pine Script, indicadores, estadísticas y seguimiento de operaciones. La afirmación amplia de gratuidad se concreta mejor consultando cada repositorio de la [organización original](https://github.com/LuxAlgo).

| Proyecto original | Función declarada | Licencia indicada por el proyecto |
| --- | --- | --- |
| [Vela](https://github.com/LuxAlgo/Vela) | Gráficos financieros interactivos | Apache-2.0 |
| [PineTS](https://github.com/LuxAlgo/PineTS) | Entorno para ejecutar lógica Pine Script | AGPL-3.0, con alternativa comercial |
| [trade-journal](https://github.com/LuxAlgo/trade-journal) | Registro de operaciones y análisis | MIT |
| [market-trackers](https://github.com/LuxAlgo/market-trackers) | Extracción de registros públicos del mercado estadounidense | MIT para el código |
| [market-trackers-data](https://github.com/LuxAlgo/market-trackers-data) | Distribuciones de datos con enlaces de procedencia | CC0-1.0 declarada para el conjunto distribuido |

Es una verificación de existencia, finalidad declarada y licencia publicada, no una auditoría funcional. El permiso del código no decide los derechos de cada proveedor conectado. La declaración CC0 del conjunto distribuido tampoco sustituye revisar restricciones que puedan corresponder a materiales de origen.

El README de [market-trackers-data](https://github.com/LuxAlgo/market-trackers-data) explica que sus archivos diarios agrupan por fecha de ingesta, no por fecha del evento, y que los más recientes pueden reescribirse. Para un backtest sería necesario preservar versiones y comprobar la disponibilidad histórica de cada registro. El nombre de un archivo no basta como marca temporal del acontecimiento.

Para MARS-TITAN, Vela y el diario pueden orientar una futura presentación de resultados. Market-trackers permite explorar fuentes complementarias para la fase 1. PineTS solo aportaría un contraste práctico si se decidiese comparar reglas técnicas. La utilidad es secundaria frente a asegurar datos temporales y evaluación reproducible. Estos proyectos no justifican una mejora de predicción ni de rentabilidad. No se han instalado ni incorporado dependencias.

## Abomination81: bot de Polymarket

Referencia: [post de @Abomination81](https://x.com/Abomination81/status/2098925882530549822), 13 de septiembre de 2026. [Consulta oficial oEmbed](https://publish.twitter.com/oembed?url=https%3A%2F%2Ftwitter.com%2FAbomination81%2Fstatus%2F2098925882530549822).

El fragmento anuncia un bot gratuito y atribuye a su autor ganancias aproximadas de 1,5 millones de dólares en los últimos años. Se ha verificado que el autor expresa esa afirmación. No se ha verificado la cifra, su cálculo, el capital expuesto ni su atribución al código publicado.

El recurso original localizado es [Abomination81/copybot](https://github.com/Abomination81/copybot), cuyo README enlaza la misma cuenta de X. Describe un ejecutor de copia de operaciones de Polymarket, con motor Rust, procesos de supervisión en Python y panel. La página es visible públicamente, aunque el README conserva referencias a una vista previa privada: esa discrepancia debe quedar registrada.

El propio README indica que no se ha elegido una licencia de código abierto para el proyecto y que no es una distribución MIT. También señala que no incluye identidades de wallets financiadas, lista de líderes ni historiales de operaciones. Por ello no se acredita permiso general para reutilizar su código ni un historial auditable de resultados. No se ha ejecutado, conectado una wallet o operado con el bot. [Fuente original y condiciones publicadas](https://github.com/Abomination81/copybot).

Su relación con el proyecto es limitada: trata ejecución y copia de operaciones en mercados de predicción, mientras que el objetivo de MARS-TITAN es estudiar predicción multimodal y retornos residuales de acciones. Puede servir como lectura de contexto sobre separación entre señal, ejecución y supervisión, sin incorporarlo como baseline de rentabilidad ni como evidencia de eficacia de aprendizaje automático.

## Estado para el repositorio

Esta revisión añade documentación y enlaces de procedencia. No supone adoptar herramientas, instalar paquetes ni iniciar código científico. Las redes sirven para descubrir material. La incorporación posterior de una fuente exigirá comprobar que resuelve una necesidad concreta y registrar versión, disponibilidad y derechos de uso. Las fuentes científicas y las condiciones de evaluación se recogen por separado en [finance-review.md](finance-review.md).
