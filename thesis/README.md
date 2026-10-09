# Documento de trabajo de MARS-TITAN

`main.tex` es una base de redacción con la pregunta, el alcance y la organización del estudio. Distingue los ensayos de referencia ya documentados de la comparación confirmatoria, todavía pendiente. Las colecciones BibTeX de `docs/references/` se enumeran en el registro `catalogs.json` y están incluidas en la fuente LaTeX.

Esta base organiza el contenido de trabajo y no se presenta como documento final. Antes de publicar una versión completa habrá que revisar su formato y confirmar las instrucciones de citación aplicables.

El estilo `biblatex-apa` se adopta como herramienta de trabajo. La edición concreta de APA deberá ajustarse a la política de citación elegida para la versión final. Para compilar se necesita una distribución LaTeX con `latexmk`, `pdflatex`, `biber`, `babel-spanish` y `biblatex-apa`.

```bash
cd thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Antes de revisar una versión publicada, su fuente se conserva en `revisions/` con la fecha y el commit de origen en el nombre. `revisions/main-20261009-0bf7fd0b.tex` es la versión publicada en `main` antes de añadir el estado técnico de la campaña desde 2000. `revisions/main-20261009-7e717cb7.tex` es la publicada después, antes de incorporar la edición verificada, los objetivos residuales, la elección de la variante A y las comprobaciones CUDA. Las copias mantienen las rutas bibliográficas relativas a `thesis/`, así que para compilarlas hay que situarlas en esa carpeta.

No se ha instalado una distribución LaTeX como parte de la preparación. La compilación se comprobará cuando esté disponible ese entorno. La documentación Markdown y el repositorio sí tienen controles ejecutables propios.
