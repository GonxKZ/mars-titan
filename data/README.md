# Datos y manifiestos

Los datos originales existentes permanecen en `dataset/`, fuera del control de versiones. Esta carpeta contiene documentación y `manifests/`. Los futuros derivados se guardarán en `raw/`, `interim/`, `processed/` y `embeddings/`, ignorados por Git.

No hay datos transformados ni conjuntos experimentales generados en esta preparación. El diseño de muestras y sus reglas se especifican en [el contrato de datos](../docs/data/data-contract.md). Un manifiesto completo se creará únicamente sobre los archivos que realmente se usen, con origen, licencia, hash y corte temporal.
