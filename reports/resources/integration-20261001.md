# Verificación de la integración del 1 de octubre de 2026

Las PR [187](https://github.com/GonxKZ/mars-titan/pull/187) y [188](https://github.com/GonxKZ/mars-titan/pull/188) se comprobaron juntas antes de integrarlas. El árbol Git verificado es `2052cf6f290103bae240d792b71d6abd01977b62`, idéntico al del commit `c031411c3be3ca646ea3892722604d56fc892c7a` de `develop`. Los cambios posteriores de esta entrega solo actualizan documentación y recibos saneados.

La ejecución local se limitó a CPU, con un trabajador de compilación y un hilo por biblioteca numérica. Las campañas permanecieron pausadas. No se ejecutaron experimentos ni validaciones científicas nuevas.

| Comprobación | Resultado observado |
| --- | --- |
| Ruff, análisis y formato | Sin errores, 493 archivos revisados por el formateador. |
| Integridad del repositorio | 837 archivos, 123 referencias y 64 tareas comprobados. |
| Pruebas del frontend con Node | 27 aprobadas. |
| Presentación en Chromium | Ocho pruebas aprobadas, con anchos desde 320 px, catálogo estable y recuperación de paginación sin duplicados. |
| Paginación con datos públicos | 20 páginas comprobadas, carga bajo demanda, fallo recuperable y vista móvil. |
| Compilación nativa C++20 Release y CTest | Ejecutores PPO y simulador compilados, 14 pruebas aprobadas en 3,09 s. |
| Suite Python con CUDA desactivada y binarios nativos disponibles | 2.330 aprobadas, 46 omitidas y 120 excluidas en 239,05 s. |
| Integraciones de escenarios y PPO con `hmmlearn==0.3.3` | 71 aprobadas en 25,51 s. Incluyen las ocho pruebas omitidas por esa dependencia en la suite anterior. |

Las 71 pruebas de la última fila se solapan con la suite general. No deben sumarse como una suite independiente. De las 120 exclusiones, 119 requieren CUDA de forma explícita y una comprueba la cola GPU. Las otras omisiones incluyen CUDA y una extracción que requiere activación explícita. No se reemplazó ningún cálculo CUDA por CPU para presentar esas rutas como verificadas.

La primera ejecución general encontró 75 fallos por ausencia del ejecutable `mars-titan-sim` y 119 por requerir CUDA. Se compiló el ejecutor que faltaba y se repitió el perfil CPU completo. Este informe corresponde a esa repetición. Los avisos de deprecación de TorchScript y NumPy siguen visibles.

Estos tiempos describen las comprobaciones locales. No son una comparación de rendimiento ni demuestran aceleración. La comprobación completa de CUDA de esta integración queda pendiente. Las evidencias CUDA anteriores conservan sus propias versiones y fechas.
