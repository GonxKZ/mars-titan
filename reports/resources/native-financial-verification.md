# Verificación del simulador C++20

Comprobación local del 24 de septiembre de 2026 en Linux x86_64. La huella de fuentes del ejecutable es `2100dde5ad52345d8552bd80abced094917f678e354507e60ca115e1e8488e4c`. El [registro de perfiles](native-financial-profiles.json) identifica cada compilación y sus tiempos incrementales con dos trabajos de construcción. Clang 21.1.8, GCC 15.2.0, clang-tidy 21.1.6, CMake 4.2.3 y Ninja 1.13.2 mantienen C++20 y el consumidor C17.

## Pruebas y diagnósticos ejecutados

| Comprobación | Resultado y alcance |
| --- | --- |
| Release y GCC Debug | Cuatro CTests y 75 pruebas del ejecutable en cada perfil. |
| ASan y UBSan | Cuatro CTests y 75 pruebas del ejecutable, con detección de fugas y sin informes de los sanitizadores. |
| TSan | Cuatro CTests y cinco casos del ejecutable, incluidos 1, 2, 4 y 8 trabajadores y recuperación con un número distinto de trabajadores. |
| MSan | Cuatro CTests del núcleo y sesión, con libc++ y libc++abi instrumentadas. |
| Análisis estático | clang-tidy, análisis de vida útil experimental compatible y analizador de rutas de Clang sobre cinco fuentes de producción. El programa de fuzzing de serialización también pasa clang-tidy. |
| Fuzzing | Siete CTests, incluidos tres programas con 1000 entradas cada uno para contabilidad, sesión y serialización. Semilla 42, entradas de hasta 4096 bytes y límite de 2048 MiB. |
| Adaptador Python | Catorce pruebas de estados, observaciones, lotes, monedas, alineamiento, excepciones de asignación y recuperación de comparadores. Usan diagnóstico CPU explícito. |

El ejecutable Release tiene SHA-256 `68b87f5eeb454705e1131b8cc5eacc25f0991d66f3fc3ea65713538a0f7eb9c9`. La inspección de enlaces confirma Arrow C++, Parquet, OpenSSL y la biblioteca contable, sin `libpython` ni `arrow_python`.

Las pruebas del ejecutable comparan las métricas de las nueve combinaciones de referencia y coste con Python. Comprueban lecturas Parquet y calendarios, rechazo del test y de históricos no admitidos, manifiestos alterados, errores numéricos, recuperación exacta, versiones de compilación incompatibles y conservación de archivos ante corrupción. Las entradas inválidas deben producir el código 1 y el mensaje controlado del programa. Una caída o un diagnóstico de sanitizador hacen fallar la prueba.

La revisión detectó un bloqueo al abrir un FIFO antes de comprobar su tipo. Las lecturas usan ahora `O_NONBLOCK` y rechazan el archivo mediante `fstat`. Otra regresión impedía confiar en la procedencia de un recibo al recuperar. Su versión, campos de identidad, estados, tiempos, fechas y metadatos del checkpoint se validan antes de escribir. Los casos corruptos conservan los bytes del recibo y del estado. Un recibo anterior al checkpoint confirmado sigue siendo admisible, porque una interrupción puede ocurrir entre ambas publicaciones.

Los avisos sobre índices y tamaños se corrigieron en el código. Las supresiones puntuales de clang-tidy se limitan a la firma variádica obligatoria de `open` y a la unión que glibc utiliza para `ru_maxrss`. Cada supresión explica el contrato POSIX o ABI que la justifica. No se desactivaron familias de comprobaciones para conseguir una compilación válida.

## Cobertura, complejidad y precisión

El [informe de calidad](native-financial-quality.json) registra 1935 de 2256 líneas (85,77 %) y 977 de 1472 ramas (66,37 %). Son los totales LF/LH y BRF/BRH exportados por LLVM 21, sobre fuentes y cabeceras propias. Excluyen pruebas, biblioteca estándar y dependencias. Las cifras proceden de los cuatro CTests y las 75 pruebas externas del ejecutable instrumentado.

Lizard 1.17.31 calcula complejidad ciclomática. CRAP utiliza `CCN² × (1 − cobertura)³ + CCN`. Para cada función se unen las líneas DA ejecutadas dentro de su intervalo. Las instancias y lambdas pueden compartir líneas. `validate_snapshot` y `MarketTape::validate` presentan los mayores valores, 125,10 y 92,75, con CCN de 76 y 62. Son validadores con muchas condiciones y conservan combinaciones sin recorrer. Estos diagnósticos no equivalen a una demostración de corrección.

LLVM emite cuatro avisos de datos incompatibles al combinar objetos. Se localizaron en las entradas con hash cero de `CashMovements::add` y `reconciled`, presentes dos veces como funciones inline sin uso en esas unidades. Los perfiles reales tienen hashes `0x60e` y `0xd7d835d`, con llamadas registradas, y la biblioteca por separado no produce ese aviso. El [lector de cobertura de LLVM 21.1.8](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/ProfileData/Coverage/CoverageMappingReader.cpp) identifica esas entradas mediante `isCoverageMappingDummy`. La [incidencia 72786 de LLVM](https://github.com/llvm/llvm-project/issues/72786) describe problemas al combinar cobertura y bibliotecas compartidas. Los avisos se conservan y no se cambió el cálculo para silenciarlos.

Las [tres mutaciones dirigidas](native-financial-mutations.json) compilaron y fueron detectadas por pruebas de comportamiento. Adelantaban un cobro del cierre a la apertura, eliminaban los costes o permitían recuperar un origen distinto. El registro identifica las sustituciones, la fuente y los fallos observados. No representa una campaña exhaustiva de mutación.

La inspección del ensamblado Release confirma sumas y restas escalares dependientes entre parciales de `AccurateSum`. La observación incluye una división SSE2 sobre dos doubles y una llamada escalar a `log1p`. No se observaron instrucciones FMA en los símbolos examinados. Esto no demuestra que todo el bucle se vectorice ni atribuye la mejora total a una instrucción. La [comparación completa](native-go-no-go.md) conserva la precisión y mide el proceso separado de los perfiles instrumentados.

## Reproducción y límites

Los [presets y sus dependencias](../../native/README.md) permiten repetir cada perfil. Para incluir las pruebas externas en cobertura, desde la raíz después de construir `native-coverage`:

```bash
LLVM_PROFILE_FILE="$PWD/build/native/native-coverage/coverage/%m-%p.profraw" \
MARS_TITAN_SIM_EXECUTABLE=build/native/native-coverage/mars-titan-sim \
  uv run --no-sync pytest -q tests/simulation/test_native_runner.py
ctest --test-dir build/native/native-coverage --output-on-failure
cmake --build build/native/native-coverage --target coverage-report
llvm-cov-21 export build/native/native-coverage/libmars_titan_simulation.so \
  -object build/native/native-coverage/simulation_tests \
  -object build/native/native-coverage/threaded_simulation \
  -object build/native/native-coverage/c_abi_test \
  -object build/native/native-coverage/financial_session_tests \
  -object build/native/native-coverage/mars-titan-sim \
  -instr-profile=build/native/native-coverage/coverage/native.profdata \
  -ignore-filename-regex='(/usr/|_deps/|[.]venv/|native/tests/)' \
  -compilation-dir=. -format=lcov > build/native/native-coverage/coverage/native.lcov
uv run scripts/summarize_native_quality.py \
  --lcov build/native/native-coverage/coverage/native.lcov \
  --output artifacts/native-financial-quality.json
```

MSan no cubre las bibliotecas precompiladas de Arrow y OpenSSL ni el ejecutable que las enlaza. No se han probado Windows, entrenamiento CUDA, históricos reales, cortes eléctricos ni cambios hostiles simultáneos del sistema de archivos. El fuzzing nuevo cubre el contrato JSON y los snapshots propios. No verifica la implementación interna de Parquet. El test final permanece cerrado.
