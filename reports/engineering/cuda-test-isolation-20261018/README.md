# Aislamiento de las pruebas CUDA

`make test-cuda` ejecutaba toda la selección en un proceso pytest. Una prueba que creaba un contexto CUDA podía hacer que los lanzadores nativos posteriores rechazasen la GPU por detectar ese mismo proceso. Las dos paridades iniciales de `native_ppo` y `native_klpo` fallaron así durante la verificación local del 18 de octubre.

El commit `d6a67e3ff4d4c56411635640cbc7273ce05cc8da` recoge los nodeids en un hijo y ejecuta cada caso en otro proceso, en serie. El supervisor usa la biblioteca estándar y el intérprete del entorno uv. Cada hijo guarda su salida, las fases de pytest y JUnit. El recibo conjunto conserva errores, omisiones, interrupciones y casos sin ejecutar. La protección de aprendizaje se hereda y sigue cerrada.

Las 36 pruebas CPU pasan. Incluyen parámetros con espacios, aislamiento de PIDs, rechazo de recibos inconsistentes, errores de colección, omisiones, timeout, señales y limpieza del hijo cuando falla una escritura. Se midieron 153/153 sentencias y 32/32 ramas de los dos módulos nuevos con coverage.py 7.16.1, incluidos sus subprocesos. Radon 6.0.1 da CRAP máximo 10 usando cobertura de sentencias. Las 13 mutaciones dirigidas se detectaron y las fuentes se restauraron con sus huellas comprobadas.

La comprobación CUDA ejecuta una inferencia financiera con un registrador de gradientes que no modifica pesos, seguida de las evaluaciones iniciales de `native_ppo` y `native_klpo`. Los tres casos pasan en la RTX 4070 Laptop GPU y usan cinco PIDs distintos entre supervisor, colección y casos. El supervisor no importa PyTorch. La ejecución tarda 34,70 segundos dentro del lanzador. El recibo conserva también el tiempo del proceso completo y el máximo de memoria residente que informa GNU time. No se ha repetido la suite CUDA completa con este lanzador.

[Recibo de verificación](verification.json), [fases de los tres casos CUDA](cuda-cases.json), [cobertura, complejidad y CRAP](quality.json) y [mutaciones dirigidas](mutations.json). La [issue #670](https://github.com/GonxKZ/mars-titan/issues/670) delimita el cambio. La orden y sus opciones se describen en [tests/README.md](../../../tests/README.md#parte-cuda).
