# Verificación y coste del análisis de campañas

Las medidas corresponden a leer resultados congelados y ejecutar controles financieros sin aprendizaje. No miden una aceleración del entrenamiento ni del futuro candidato MARS-TITAN. El [informe científico](../baselines/campaign-comparison-20261005.md) recoge las conclusiones y sus límites.

## Recorrido predictivo

La primera lectura de 1.512 Parquet verificó sus hashes y reconstruyó MAE y MSE en 10,964 segundos, con 8,123 segundos en lectura y 1,406 en métricas. Es una referencia de trabajo limitado, sin IC, política inicial ni intervalos. No se compara como si hiciera lo mismo que el análisis final.

El perfil completo con `cProfile` empleó 53,049 segundos. La reconstrucción de 1.056 políticas iniciales acumuló 8,303 segundos. El cálculo era idéntico entre muchos ajustes del mismo padre, por lo que se añadió una caché de entradas y valores acotados. La lectura Parquet sigue utilizando Arrow y las operaciones numéricas NumPy y SciPy. No se añadió otro lector ni un kernel propio.

El [benchmark completo](campaign-comparison-20261005-performance.json) compara la misma implementación con caché desactivada y activada. Alterna el orden, ejecuta un calentamiento por condición y tres repeticiones medidas. Incluye arranque, lectura, hashes, diagnósticos, remuestreo y escritura. Las tablas científicas y el Parquet son idénticos byte a byte. Los casos también coinciden al excluir su tiempo de comprobación.

| Condición | Tiempo mediano del proceso | Rango de tres repeticiones | RAM máxima del proceso |
| --- | ---: | ---: | ---: |
| Sin caché | 46,965 s | 45,640–47,265 s | 359,11 MiB |
| Caché de hasta 8 MiB | 37,455 s | 37,209–37,589 s | 362,03 MiB |

La reducción de la mediana es del 20,25 %, un factor de 1,254. Se reconstruyen 48 políticas y se reutilizan 1.008 resultados. Al terminar, la caché retiene 4.838.928 bytes. Su clave incluye el vector completo de predicciones y la rejilla, por lo que no reutiliza predicciones sobre entradas diferentes.

El perfil posterior empleó 45,149 segundos bajo instrumentación. Las llamadas de correlación acumularon 10,865 segundos y la lectura Arrow 9,251. Estos tiempos incluyen cálculo y control asociados a esas llamadas. No permiten atribuir todo el tiempo de lectura a disco físico ni demostrar que se haya alcanzado el límite del hardware. No se sustituyeron las primitivas existentes basándose solo en esos contadores.

Equipo y versiones: Ryzen 9 8945HS, Linux x86-64, Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1 y PyArrow 25.0.1. OpenBLAS y OpenMP se limitaron a un hilo. Las fuentes ya se habían leído y no se detuvieron aplicaciones ajenas. Los 947.946.963 bytes corresponden a archivos de predicciones verificados, no a tráfico físico medido. No se midieron energía ni coste monetario. La ruta de análisis no usa GPU.

## Controles financieros nativos

El ejecutable C++20 evalúa 512 mundos, seis políticas fijas y tres costes. Cada ejecución termina 9.216 episodios, sin episodios inválidos, y 2.350.080 transiciones. Incluyen 589.824 transiciones de calentamiento. Se leen 86.507.016 bytes de Parquet de mercado, con 2.097.152 filas de activo y sesión.

Después de la revisión se ejecutó un calentamiento y cinco repeticiones. La mediana fue 3,067 segundos, con rango de 3,006 a 3,439, desviación muestral de 0,180 segundos y máximo de 59.056.128 bytes de RAM. Los resultados coinciden exactamente con los anteriores a la corrección de rutas. Esa corrección no se presenta como una optimización. El [recibo nativo](campaign-comparison-20261005-native.json) conserva fuentes, binario, hashes autorizados, pruebas y medidas.

Se usó un trabajador. No se copian modalidades por política y cada cinta de mercado se reutiliza para las dieciocho combinaciones de regla y coste. El ejecutable no enlaza Python, LibTorch ni CUDA. No se han medido throughput energético ni contadores de hardware.

## Correctitud y revisión

Las pruebas nuevas cubren fechas reservadas antes de leer etiquetas, identidades y hashes, rejillas de otra cohorte, duplicados de muestras y semillas, cambios de población, orden de activos, métricas conocidas, valores no finitos, límites de trabajo, RNG, truncamiento de bloques y conservación de una salida anterior. La comparación financiera añade pruebas de costes, mundos ausentes, selección, calentamiento y media previa de semillas.

La revisión encontró y corrigió tres defectos: faltaba comprobar la procedencia de la rejilla, las rutas relativas podían confundir los límites de las fuentes en C++, y el exportador necesitaba un estado de terminación explícito en el recibo financiero. Las regresiones fallaron antes de cada corrección. Las figuras SVG y PNG se inspeccionaron y se corrigió un solapamiento entre la etiqueta y la nota de la figura predictiva.

Pasan las 27 entradas CTest del perfil nativo Release. Las tres entradas específicas también pasan con clang-tidy y Clang Static Analyzer, y con AddressSanitizer y UndefinedBehaviorSanitizer. Se usó Clang 21.1.8, C++20, libstdc++ y el análisis experimental de Lifetime Safety que admite el perfil. Arrow y OpenSSL son bibliotecas precompiladas y no se presentan como instrumentadas por esos sanitizadores.

SIGINT se comprobó con un catálogo indicado mediante `index.json` desde su propio directorio. El recibo quedó interrumpido, con 396 episodios correspondientes a 22 mundos completos. Una salida relativa dentro del catálogo se rechaza antes de crearla.

## Cobertura y límites

La [medición Python](campaign-comparison-20261005-quality.json) utiliza coverage.py 7.16.2 con ramas, Radon 6.0.1 y pytest 9.1.1 sobre los cuatro módulos de análisis. Pasan 73 pruebas dirigidas, con 819 de 843 sentencias cubiertas (97,15 %) y 197 de 212 ramas (92,92 %). CRAP se calcula como `CC² × (1 − cobertura de sentencias)³ + CC`, dentro del intervalo de cada función de Radon. Las figuras y los scripts de medición se ejercitan mediante ejecuciones completas y revisión visual, fuera de ese porcentaje.

`aggregate_results` y la admisión financiera concentran complejidad ciclomática 56 y 55, aunque sus sentencias están cubiertas. Esos valores señalan funciones que merecen atención al ampliarlas, no ausencia de errores. Las mutaciones dirigidas comprueban ponderación por sesión, población, normalización del bootstrap, reserva temporal, procedencia de la rejilla, media de semillas, hashes y calentamiento. No equivalen a mutación exhaustiva.

LLVM 21 y Lizard 1.17.31 miden el código nativo. `financial_controls.cpp` alcanza 83,10 % de líneas y 62,11 % de ramas. La comprobación de rutas `contains` tiene todas sus líneas cubiertas y CRAP 6. Las ejecuciones completas de los 512 mundos se hicieron en Release y no se suman a la cobertura instrumentada. LLVM conserva un aviso sobre entradas con hash cero de dos métodos anteriores de `CashMovements`. No se ocultó el aviso ni se presenta una cobertura global de ese núcleo.

Las nuevas rutas son de CPU. La comprobación del entorno confirmó RTX 4070 Laptop, controlador 595.91.07, PyTorch 2.14.0+cu130 y CUDA 13.0 disponible. Dos pruebas de interoperabilidad entre boosting y PyTorch CUDA pasaron. Eso no se utiliza para afirmar una nueva medición CUDA del análisis ni entrenamiento del candidato.

La suite general terminó con 2.819 pruebas correctas, 19 omitidas y ocho avisos en 332,51 segundos. Se ejecutó con CUDA visible y las tres rutas explícitas de biblioteca, simulador y PPO del perfil conjunto. Once omisiones corresponden a integraciones que requieren una ventana CUDA exclusiva y activación explícita. Las otras ocho necesitan `hmmlearn`, ausente del entorno. No se han modificado esas rutas ni sincronizado el entorno científico para instalarla. Los avisos proceden de límites infinitos de espacios Gymnasium y de la futura retirada de funciones TorchScript, sin ocultarlos.

Ruff, su comprobación de formato y `scripts/check_repository.py` pasan. La comprobación del repositorio recorre 978 archivos, 191 referencias y 64 tareas. Los CSV, SVG, PNG y su índice se regeneraron con igualdad exacta y se comprobaron sus hashes. No se publican pesos, trayectorias por mundo, etiquetas individuales ni rutas privadas.
