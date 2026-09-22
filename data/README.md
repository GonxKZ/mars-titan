# Datos y manifiestos

Los datos originales existentes permanecen en `dataset/`, fuera del control de versiones. Esta carpeta contiene documentación y `manifests/`. Los futuros derivados se guardarán en `raw/`, `interim/`, `processed/` y `embeddings/`, ignorados por Git.

Los datos transformados y los conjuntos experimentales generados permanecen fuera de Git. El diseño de muestras y sus reglas se especifican en [el contrato de datos](../docs/data/data-contract.md). Cada manifiesto identifica los archivos realmente utilizados, con origen, licencia, hash y corte temporal.
