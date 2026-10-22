# Coste del archivo de trazas regionales

El perfilado de `shared_evidence_bridge_v1` localizó un coste alto en la comprobación del tamaño JSON y en la preparación de artefactos CPU. Se ha cambiado la comprobación de tamaño de `mechanism_history.py`. El cálculo de las vistas, los puentes, la selección episódica, MAC y la cabeza conservan sus operaciones.

La validación recorre los tipos ordinarios, profundidad, número de elementos, enteros de 64 bits, finitud y longitud de cadenas. Ahora también devuelve una cota superior conservadora del JSON ASCII. Cuenta hasta doce bytes por carácter Unicode, incluidos pares de sustitutos, y acota números y separadores. Si esa cota cabe en el límite declarado, utiliza el codificador completo de la biblioteca estándar. Si no basta para decidir, conserva el recorrido incremental anterior y comprueba el tamaño exacto. Una cota demasiado conservadora no rechaza por sí sola un evento válido. No se han aumentado los límites ni cambiado el formato del archivo.

## Medidas

El [microbenchmark](cost.json) compara veinte repeticiones sobre el mismo evento, con tres de calentamiento y orden alternado. La medida de asignaciones se ejecutó aparte con `tracemalloc`. Los valores son medianas y picos de asignaciones Python adicionales durante la llamada.

| Evento | Bytes JSON | Tiempo anterior | Tiempo actual | Pico Python anterior | Pico Python actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lector anterior | 54.510 | 5,42 ms | 3,11 ms | 6.360 B | 65.536 B |
| Lector regional | 807.181 | 97,88 ms | 57,91 ms | 8.536 B | 915.804 B |

La ruta rápida intercambia memoria transitoria por tiempo. La cota limita el tamaño de la representación materializada, no certifica la RAM total del proceso ni todos los temporales del codificador. El recorrido incremental sigue disponible cuando la cota no cabe.

También se midió ese recorrido incremental con el límite fijado al tamaño exacto del JSON. En tres fixtures adicionales, las medianas anteriores y actuales fueron 14,031 y 14,050 ms para 16.384 ceros, 0,144 y 0,162 ms para cadenas ASCII y 0,104 y 0,114 ms para Unicode fuera del BMP. La cota añade trabajo en estos casos. No se afirma que la optimización acelere cualquier entrada.

La comparación de la sesión nativa usa la misma fuente preparada, IDs de muestras, parámetros iniciales, banco, semillas y configuración. Cada versión ejecuta los dos lectores con 64 filas, ruta `k4cr`, planificación y bloques físicos de 64 filas, dimensión 32, tres regiones, anillo y dos ciclos. Utiliza FP64 en `cuda:0`, sin TF32 ni autocast, una repetición de calentamiento y tres medidas. Incluye emisión, comprobaciones y archivo. El settlement siguiente se mide aparte.

| Lector | Decisión anterior | Decisión actual | Settlement anterior | Settlement actual |
| --- | ---: | ---: | ---: | ---: |
| Anterior | 2.555,15 ms | 1.973,26 ms | 1.873,09 ms | 856,24 ms |
| Regional | 4.505,37 ms | 3.871,29 ms | 1.308,30 ms | 1.242,67 ms |

Son dos pasadas consecutivas sin contrabalancear su orden. Otra aplicación permaneció activa, con un 85 % y un 93 % de utilización GPU observados antes de las pasadas. No se detuvo. No se atribuye todo el cambio de latencia al serializador. Las muestras y sus percentiles están en el resumen, pero tres repeticiones no caracterizan las colas. El resultado no se extrapola a 256 filas ni a datos reales.

El pico asignado por PyTorch fue idéntico antes y después, 76.191.232 bytes para el lector anterior y 76.290.048 para el regional. Es memoria de este proceso, no VRAM total. El pico RSS de proceso fue de 1.631.032 y 1.630.180 KiB. La diferencia pequeña de RSS no acredita ahorro de memoria. Los procesos duraron 148,45 y 108,72 segundos externos, con importación, preparación y calentamiento incluidos. No se midió energía.

Los datos son un fixture con el mismo historial de precios replicado por identificador y un canal gráfico presente constante de valor 3. Noticias, fundamentales y macro están ausentes con sus máscaras. Los 64 identificadores no son 64 historias financieras independientes. Las etiquetas de feedback son manuales. No hubo pasos de optimizador, entrenamiento ni evaluación predictiva.

## Comprobaciones y procedencia

Las [59 pruebas de historial y límites](verification.json) y las 43 de integración pasaron sin fallos, errores ni omisiones. Nueve casos de integración están marcados CUDA. Comprueban también recuperación de la siguiente predicción y actualización tras un corte confirmado, selección automática y aplicación única del feedback. Esta selección no equivale a toda la suite del proyecto.

Los [13 mutantes dirigidos](mutations.json) fueron detectados, con 24 pruebas en la referencia inicial y otras 24 al restaurarla. Incluyen subestimar Unicode y números, omitir límites, aceptar no finitos o enteros fuera de rango, materializar sin cota, devolver la cota como tamaño exacto y omitir el límite del recorrido incremental. La prueba se reformateó después con Ruff. Su AST coincide con el usado en mutación y las 59 pruebas se repitieron sobre el archivo final.

La [cobertura y CRAP](quality.json) usa coverage.py 7.16.1, Radon 6.0.1 y `scripts/summarize_python_quality.py`. Las dos funciones modificadas cubren todas sus sentencias y 30/30 ramas, con CRAP 17 y 4. El módulo completo registra 220/249 sentencias y 91/114 ramas. CRAP utiliza cobertura de sentencias y publica las ramas por separado. Estos diagnósticos no demuestran ausencia de errores.

La [paridad](parity.json) compara ocho parejas de sesiones, incluidas las de calentamiento de la medida. Los valores de predicción, estado rápido, pesos, vistas, episodios y demás contenido contrastado coinciden exactamente. Las huellas de código cambian la identidad de ejecución y, por derivación, los IDs de predicción, modelo, contexto y snapshot. Se conservan esas diferencias en los archivos. No hay identidad byte a byte entre sesiones. El comprobador separa solo esos campos de procedencia y exige igualdad en el resto.

El [perfil de funciones](profile.json) procede de `cProfile` con el código anterior. Sus tiempos instrumentados permiten localizar trabajo CPU. No son medidas de kernels GPU y sus tiempos acumulados no forman categorías sumables. El primer intento de instrumentación falló por colisión con un módulo de Python y por la protección contra sustitución de métodos. Se conserva como fallo y no aporta resultados. El intento válido envuelve las llamadas desde el ejecutable y mantiene los métodos originales.

La referencia es el código de `ed0e084fdc57b3dc1efb93a0f5d4c872020d7250`. Se comprobó además que el bytecode importado en la pasada de referencia coincide con esa fuente. El cambio actual y la referencia están fijados por SHA-256. Los scripts exactos, XML, perfiles, protocolos, fuentes y artefactos nativos se identifican en [external-artifacts.json](external-artifacts.json) y permanecen fuera de Git.

El ejecutable de coste del controlador sigue disponible para una nueva medida independiente con fixtures:

```bash
evidence_dir="$(mktemp -d)"
mkdir "$evidence_dir/storage"
uv run --no-sync python reports/engineering/bridge-controller-20261021/benchmark.py --output "$evidence_dir/cost.json" --storage-root "$evidence_dir/storage" --rows 64 --routes k4cr --reader-max-batch 64
```

Esta orden crea otra fuente y no sustituye la pareja histórica fijada. La reconstrucción de esa pareja requiere los artefactos externos y sus scripts archivados. El puente sigue siendo opcional y permanece desactivado por defecto. No hay mejora predictiva medida ni cambio en el bloqueo de aprendizaje o el test final.
