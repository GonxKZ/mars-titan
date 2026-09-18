# Posibles kernels CUDA

Este directorio está reservado para kernels que puedan justificarse con un perfil del modelo. Actualmente no contiene código CUDA científico.

La primera opción será usar operaciones disponibles en PyTorch sobre `cuda:0`. Un kernel propio requerirá una comparación con la referencia que incluya resultados, gradientes cuando proceda, tipos numéricos, formas, memoria y casos límite. También se medirá el coste total, incluido el intercambio de datos.

Activar `MARS_TITAN_ENABLE_CUDA` solo comprueba la configuración de las herramientas. No prueba una operación del proyecto. En el equipo registrado, el toolkit es 13.4 y el runtime de PyTorch es 13.0. Esa diferencia debe revisarse antes de compilar extensiones, junto con el compilador anfitrión y la ABI.

Si CUDA no está disponible o aparece un error de compatibilidad, la ejecución deberá detenerse con el motivo registrado. El paso a CPU exige una decisión explícita y una descripción del cambio de condiciones del experimento.
