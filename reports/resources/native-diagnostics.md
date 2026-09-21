# Verificación local de herramientas nativas

Fecha: 21 de septiembre de 2026. Alcance: configuración CMake y programas C++
temporales de prueba. No se ha implementado ni medido un kernel científico.

## Entorno y comprobaciones

Se usaron CMake 4.2.3, Clang 21.1.8 y clang-tidy 21.1.8 en Linux x86_64.
Los perfiles de configuración también se generaron con GCC 15.2.0. clang-tidy y
clang-format se obtuvieron de paquetes oficiales, sin modificar la instalación
del sistema ni incorporar ejecutables al repositorio.

Las siete pruebas de `tests/native/test_diagnostics.py` comprueban compilación
C++20 sin extensiones, avisos como errores, un acceso fuera de límites real,
una desreferencia nula, separación entre sanitizadores y perfilado, y la ausencia
de herramientas tanto con objetivos como sin ellos. El fallo de memoria exige
la salida de AddressSanitizer y código 1. Una terminación por señal no cuenta
como detección válida.

La prueba de configuración sin objetivos detectó que se aceptaba una ruta
inexistente de clang-tidy. La validación se trasladó al momento de configuración
y el caso pasó después de fallar antes del cambio.

```bash
MARS_TITAN_CLANG_TIDY=/ruta/a/clang-tidy \
  CUDA_VISIBLE_DEVICES=-1 uv run --locked --extra research --extra cuda \
  --extra encoders pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_repository.py
```

La suite comprobada registró 429 pruebas correctas y 16 omitidas explícitamente
por requerir CUDA. CUDA se ocultó para esta comprobación de herramientas CPU,
no se sustituyeron entrenamientos acelerados por entrenamientos CPU. También
pasaron Ruff y la validación documental.

## Límites

Los perfiles `native-debug`, `native-asan` y `native-profile` se configuraron.
clang-format aceptó la configuración de estilo. No hay todavía fuentes nativas
científicas a las que aplicar cobertura, complejidad o mutación. Las pruebas
de fallos conocidos comprueban los diagnósticos, no la corrección de algoritmos
que aún no existen.

ThreadSanitizer está configurado, pero no se ejecutó sobre código concurrente.
Tampoco se ejecutaron Nsight, Compute Sanitizer ni Valgrind sobre una carga
científica. `perf stat -- true` terminó, pero informó contadores de CPU no
soportados. No se cambiaron permisos del kernel. Tener las herramientas
instaladas no demuestra que estén habilitados todos los contadores del equipo.

El [contrato nativo](../../native/README.md) detalla los siguientes pasos.
Una optimización futura necesita entradas y versión fijadas, paridad numérica,
medición del recorrido completo y una mejora repetible de tiempo o memoria.
