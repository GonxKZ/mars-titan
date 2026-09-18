# Biblioteca y estado del arte

La selección inicial reúne **38 referencias bibliográficas únicas** sobre datos financieros, memoria neural, evaluación, incertidumbre y fundamentos. Se han descargado **28 PDF** desde editoriales, repositorios académicos o enlaces ofrecidos por sus autores. La biblioteca incluye cinco libros completos. Otros libros se consultan en HTML o requieren acceso editorial.

La revisión tiene alcance documentado y fecha de corte del 18 de septiembre de 2026. No se presenta como una búsqueda sistemática exhaustiva de toda la literatura ni como lectura íntegra de todos los libros descargados. Su objetivo es sostener las decisiones iniciales y señalar qué lectura detallada necesita cada experimento.

## Contenido

| Área | Síntesis | Metadatos | Bibliografía |
| --- | --- | --- | --- |
| Finanzas cuantitativas y evaluación | [Revisión financiera](finance-review.md) | [finance-sources.json](finance-sources.json) | [finance.bib](finance.bib) |
| Memoria, multimodalidad e incertidumbre | [Revisión científica](neural-review.md) | [neural-sources.json](neural-sources.json) | [neural.bib](neural.bib) |
| Libros y fundamentos | [Lectura dirigida](books-review.md) | [book-sources.json](book-sources.json) | [books.bib](books.bib) |
| Posts aportados | [Finanzas y herramientas](social-finance.md), [arquitectura y rendimiento](social-neural.md) | URL y resultado de acceso dentro de cada nota | Solo se incorporan a la base científica las fuentes primarias que correspondan. |

Las reseñas de memoria incluyen ACI y EnbPI, cuyas fichas canónicas están en el catálogo financiero. Hyndman también tiene su entrada canónica en finanzas. Se evita duplicar identificadores al descargar o citar.

## Búsqueda y selección

Se partió del documento inicial, de FinMultiTime y de los conceptos necesarios para evaluar una memoria financiera. Se siguieron citas hacia autores, actas de congresos, arXiv, revistas, SEC, la biblioteca de Kenneth French y publicaciones de AQR. Los libros se localizaron en las webs de sus autores y editoriales. Los seis posts aportados se contrastaron por separado.

Familias de consulta utilizadas:

- FinMultiTime, disponibilidad temporal, noticias y tablas financieras.
- Titans, test-time training, MIRAS, ATLAS, SEAL y memoria a distintas escalas.
- Referencias compactas para series temporales y modelos multimodales financieros.
- Retornos residuales, factores, sesgo de exclusión y universo histórico.
- Overfitting de backtests, selección múltiple, DSR, Sharpe y costes de ejecución.
- Calibración, intervalos conformales, dependencia temporal y cambio de distribución.

Se priorizan fuentes primarias con autoría y versión verificables. Se distinguen artículos revisados por pares, preprints, documentación institucional, libros y divulgación. Se excluyen promesas de rentabilidad no comprobadas, copias sin procedencia, bibliografías automáticas con enlaces erróneos y material cuya descarga exige eludir controles.

La selección debe actualizarse antes de cerrar el marco teórico y antes del test final. Cada nueva referencia debe cubrir una decisión, un método o una limitación concreta. Para convertir esta revisión dirigida en una revisión sistemática formal harían falta bases, ecuaciones de búsqueda, resultados por consulta y cribado completo registrados con otro protocolo.

## Descargas y conservación

```bash
uv sync --locked
uv run python scripts/fetch_references.py
```

Los PDF se guardan en `docs/references/library/`, con nombres basados en identificadores bibliográficos. Esa carpeta está ignorada por Git. El script conserva archivos existentes, comprueba cabecera y marcador final, contrasta la longitud declarada y calcula SHA-256. Junto al PDF guarda un recibo de procedencia con URL, fecha y hash. Si cambia la URL o el archivo, la discrepancia se comunica y no se sobrescribe. Un archivo previo sin recibo se marca como `cached_unverified`, sin atribuirle la URL actual. Un límite de 128 MiB por documento evita descargas inesperadamente grandes.

El [manifiesto de esta preparación](download-manifest.json) registra los 38 resultados, incluidos documentos disponibles solo como referencia. `cached` indica que el PDF ya estaba descargado al repetir la comprobación. La inspección con `pdfinfo` comprueba además que los archivos se pueden interpretar como PDF. Un hash acredita identidad de bytes, no la validez científica ni la licencia de un documento.

El historial de acceso distingue la versión consultada de la publicada. Por ejemplo, DLinear se cita como artículo AAAI de 2023, pero el PDF de consulta es el preprint arXiv v3 de 2022 porque la descarga editorial cerró la conexión. Los catálogos conservan esa diferencia.

## Derechos y uso académico

El acceso gratuito no implica permiso para redistribuir. Los libros comerciales y artículos restringidos permanecen como fichas con su enlace oficial. Los PDF no se suben al repositorio y mantienen sus derechos originales. La bibliografía está preparada para su integración en la memoria, pero cada cita final debe corresponder a una lectura pertinente y a una afirmación que la fuente sostenga.

Para cada lectura detallada, usar la [ficha de literatura](../../reports/literature-note.md). Registrar páginas, tarea original, datos, validación, resultados realmente publicados, límites y decisión que cambia en MARS-TITAN. No trasladar rendimientos de un paper al proyecto como si se hubieran reproducido.
