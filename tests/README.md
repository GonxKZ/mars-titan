# Pruebas

Las pruebas se agrupan por la responsabilidad que comprueban:

| Directorio | Alcance |
| --- | --- |
| `tests/data/` | Disponibilidad temporal, inventario, preparación y materialización de datos |
| `tests/models/` | Entradas comunes, referencias predictivas, persistencia y reanudación |
| `tests/native/` | Configuración de CMake, avisos, análisis estático y sanitizadores |
| `tests/tooling/` | Biblioteca documental, fuentes públicas, exportador y observatorio |

Las pruebas de herramientas no acceden a la red. Comprueban identificadores, metadatos, integridad, formatos, límites de descarga y conservación de archivos anteriores. Las pruebas científicas usan datos sintéticos o artefactos locales controlados y distinguen las medidas de coste de los resultados predictivos.

Desde la raíz del repositorio se puede ejecutar el conjunto completo:

```bash
uv run --locked pytest
```

Las pruebas que requieren CUDA comprueban su disponibilidad y no cambian a CPU de forma silenciosa. Para ejecutar solo las comprobaciones compatibles con CPU se deben seleccionar los módulos correspondientes, por ejemplo:

```bash
uv run --locked pytest tests/tooling tests/native
```

Cada prueba protege una propiedad concreta. La cobertura y las pruebas de mutación ayudan a localizar lógica poco comprobada, pero no sustituyen los casos de comportamiento ni acreditan por sí solas la reproducibilidad de un entrenamiento.

## Protección del aprendizaje

Mientras exista la protección local `~/.local/state/mars-titan/training-hold-2000.json` con `training_allowed: false`, `tests/conftest.py` registra un gancho global previo a cada paso de optimizador de PyTorch. La prueba que intente ese paso se omite antes de modificar pesos o estados del optimizador, con el motivo «Bloqueo de aprendizaje vigente». La variable `MARS_TITAN_TRAINING_HOLD` permite indicar otra ruta. Sin protección, o con `training_allowed: true`, el gancho no se instala y las pruebas se comportan como antes. Un valor ambiguo del campo detiene la sesión de pruebas. Los optimizadores nativos de LibTorch en C++ no pasan por este gancho, así que sus pruebas siguen requiriendo una selección explícita.
