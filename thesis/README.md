# Documento de trabajo de MARS-TITAN

`main.tex` es una base de redacción con la pregunta, el alcance y la organización del estudio. Distingue los ensayos de referencia ya documentados de la comparación confirmatoria, todavía pendiente. Las colecciones BibTeX de `docs/references/` se enumeran en el registro `catalogs.json` y están incluidas en la fuente LaTeX.

Esta base organiza el contenido de trabajo y no se presenta como documento final. Antes de publicar una versión completa habrá que revisar su formato y confirmar las instrucciones de citación aplicables.

El estilo `biblatex-apa` se adopta como herramienta de trabajo. La edición concreta de APA deberá ajustarse a la política de citación elegida para la versión final. Para compilar se necesita una distribución LaTeX con `latexmk`, `pdflatex`, `biber`, `babel-spanish` y `biblatex-apa`.

```bash
cd thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

No se ha instalado una distribución LaTeX como parte de la preparación. La compilación se comprobará cuando esté disponible ese entorno. La documentación Markdown y el repositorio sí tienen controles ejecutables propios.
