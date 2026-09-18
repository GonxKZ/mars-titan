# Memoria de trabajo de MARS-TITAN

`main.tex` es una base de redacción con la pregunta, el alcance y la organización del estudio. Mantiene explícito que aún no hay resultados. La fuente bibliográfica son los tres archivos BibTeX de `docs/references/`.

La guía exige usar la plantilla del aula, que no se ha aportado. Esta base no se presenta como plantilla oficial ni como documento listo para depósito. Antes de una entrega habrá que trasladar el contenido, revisar formato y confirmar las instrucciones de citación.

El estilo `biblatex-apa` se adopta como herramienta de trabajo. La edición concreta de APA deberá ajustarse a las instrucciones de la titulación. Para compilar se necesita una distribución LaTeX con `latexmk`, `pdflatex`, `biber`, `babel-spanish` y `biblatex-apa`.

```bash
cd thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

No se ha instalado una distribución LaTeX como parte de la preparación. La compilación se comprobará cuando esté disponible ese entorno o la plantilla oficial. La documentación Markdown y el repositorio sí tienen controles ejecutables propios.
