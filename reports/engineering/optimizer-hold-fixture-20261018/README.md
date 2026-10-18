# Fixture de la guarda de aprendizaje nativa

El 18 de octubre de 2026, la batería de #662 detectó un fallo en `test_a_missing_label_a_new_stepping_test_or_an_open_preset_is_reported`. Se reprodujo en CPU tanto sobre `932d8ab02` como sobre `develop` `163c21294`.

El fixture copiaba `cmake`, `src`, `include` y `tests`, pero omitía `native/benchmarks`. El registro de CMake ya incluía `cuda_lifetimes_gemm`, cuya fuente es `benchmarks/cuda_lifetimes.cpp`. La comprobación de fuentes abortaba antes de que la prueba pudiera verificar sus cambios deliberados en las etiquetas y presets. Es una copia incompleta del fixture, sin relación con la divergencia numérica de #662.

La corrección de #567 añade `benchmarks` a esa copia temporal. Conserva las comprobaciones de fuentes existentes, etiquetas, lanzador y exclusión de los presets. No cambia el bloqueo de aprendizaje ni las configuraciones nativas.

Todo `tests/native/test_optimizer_steps_hold.py` pasa en CPU: 14 pruebas, sin fallos ni omisiones, en 0,73 segundos según pytest. La prueba de configuración crea un proyecto CMake sin lenguajes y no compila ni ejecuta pruebas nativas. La protección global conservó su huella. La batería general de `932d8ab02` se documenta por separado y no se ha repetido para esta corrección del fixture.

Se detectaron cuatro mutantes sobre copias en memoria del auditor: omitir de nuevo `benchmarks`, inventar etiquetas, ignorar llamadas que propagan pasos de optimizador y fingir una exclusión inexistente en el preset. La fuente del repositorio no se modificó para ejecutarlos.

La cobertura del auditor y sus pruebas es de 286/291 sentencias y 99/104 ramas, con coverage.py 7.16.1. El fixture modificado alcanza 7/7 sentencias y 4/4 ramas. Su complejidad ciclomática y CRAP son 3. El máximo CRAP del archivo es 22,0246 en `Registry._command`. Se calcula con Radon 6.0.1 y `scripts/summarize_python_quality.py`, usando cobertura de sentencias en `C² × (1 − cobertura)³ + C`. Estas medidas no son cobertura del código C++ ni del predictor.

Los recibos son [verification.json](verification.json), [quality.json](quality.json) y [mutations.json](mutations.json). No se ha utilizado GPU, ejecutado aprendizaje ni abierto una campaña científica.
