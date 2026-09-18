# Diseño de la base de investigación

Fecha: 18 de septiembre de 2026. Autor: Gonzalo García Lama.

## Alcance de esta entrega

Preparar un repositorio privado de investigación, documentación verificable, biblioteca local de consulta y herramientas de mantenimiento. La implementación y evaluación de MARS-TITAN corresponden al trabajo posterior. No hay todavía resultados experimentales que permitan afirmar mejoras.

## Decisiones

Se adopta un monorrepositorio con distribución Python `src`, configuración declarativa, documentación Markdown, bibliografía BibTeX y una memoria de trabajo en LaTeX. Los módulos se separan por responsabilidad. Una organización por fases duplicaría código. Separar cada componente en un repositorio aumentaría el coste de coordinación de un proyecto individual.

Los datos originales permanecen en `dataset/`, que ya existía y ocupa aproximadamente 109 GiB según `du -sh`. Las tablas derivadas irán en `data/`. Los datos y artefactos grandes no se versionan con Git. Los PDF académicos y las descargas de terceros tienen copias locales, catálogo, URL y SHA-256. La propuesta propia se conserva como antecedente.

Se elige licencia MIT para el trabajo original del repositorio, sin relicenciar datos ni publicaciones ajenas. GitHub almacena código, documentos de diseño, referencias y manifiestos. El repositorio se inicia privado. Cualquier difusión de la memoria deberá considerar las condiciones de originalidad y publicación de UNIR.

Python cubre el desarrollo científico. CMake prepara C/C++ y CUDA optativa. No se crea una implementación nativa antes de detectar un cuello de botella. El entorno de calidad debe funcionar sin GPU. Entrenamientos y pruebas de aceleración requieren CUDA explícita.

## Entregables comprobables

1. Los ocho indicadores de la rúbrica tienen peso, evidencia esperada y estado.
2. Cada objetivo tiene criterios de aceptación, riesgos y artefactos de salida.
3. El protocolo distingue predicción, adaptación de memoria, selección y evaluación final.
4. Cada publicación descargada conserva origen, hash y estado de acceso. No se afirman derechos no verificados.
5. El README es legible en GitHub y contiene diagramas de diseño, sin curvas de rendimiento inventadas.
6. Las comprobaciones locales y de GitHub son reproducibles desde un clon limpio.
