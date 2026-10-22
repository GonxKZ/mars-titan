# Admisión por error maduro en el controlador

El controlador ya admite M2 y M3 con el banco de tres índices existente. M2 conserva episodios según el error absoluto de la predicción realmente emitida cuando madura su resultado. M3 añade la anomalía y la relevancia de las entradas de esa misma decisión. El enlace reutiliza `TitansBinding` y conserva las huellas de `write_policy` y `write_scores`. No crea otro banco por región ni cambia los pesos rápidos de Titans mediante etiquetas financieras.

`FinancialSession` también comprueba el tramo declarado de las escalas M3 antes de crear archivos. Su fin no puede superar el inicio de decisiones de validación, calibración o evaluación. En una fase de entrenamiento preparada puede llegar al final de sus decisiones. El límite es exclusivo y no rechaza la igualdad. Esta comprobación no autentica el ajuste ni acredita cuándo se recibieron sus etiquetas. Las escalas de estas pruebas son manuales. La escala del error del diseño existente procede del pronóstico residual cero, no de un reajuste para cada controlador.

## Pruebas y precisión

La [regresión](verification.json) pasó 133 casos. Los complementos añaden dos casos de precisión y cuatro de contexto PT4 con calibración PT2 online. Se contrastaron sus identificadores y son conjuntos disjuntos. El total es de 139 casos, 36 CUDA, sin fallos, errores ni omisiones. No es toda la suite del proyecto.

Los casos nuevos comparan M2/M3, lector anterior/regional, acción fija/política con coeficientes dados y recogida opcional. El control utiliza el lector fijo equivalente, con los mismos parámetros, filas, retención, semilla y calendario. Coinciden los pronósticos y bancos comprobados. Las pruebas atan los errores a las emisiones nativas, comprueban los rasgos originales de las decisiones y rechazan una emisión antigua sustituida. Las consultas regionales solo contienen IDs del snapshot anterior al feedback.

Los cortes antes y después de confirmar la generación recuperan el mismo estado y no repiten ofertas. La recuperación con el mismo bloque exige igualdad exacta. Cambiar de cuatro filas por bloque a una mantiene la tolerancia FP64 anterior de `1e-12`. El [diagnóstico CPU/CUDA](partition-diagnostics.json) registró diferencias de hasta `1,11 × 10⁻¹⁶`, sin cambios de IDs, índices ni bytes de los bancos. Las diferencias numéricas aparecen en pronósticos, pesos rápidos y algunos componentes M3. No se han añadido redondeos a producción para ocultarlas.

El complemento PT4/PT2 comprueba emisiones, feedback de calibración, banco y cierre en la misma generación. Su primera tanda falló porque el test comparaba diccionarios con tensores mediante `==`. Se sustituyó ese comparador por la comprobación recursiva exacta ya utilizada en el proyecto. No se modificó producción. Las etiquetas y fuentes son fixtures con noticias y fundamentales presentes solo en los flujos indicados por sus máscaras.

La [comprobación posterior del colector](precision-check.json) usa 64 filas y 256 episodios en CPU y CUDA. Compara directamente los flotantes binary64 del JSON, sin crear tensores FP32 intermedios. Conserva igualdad exacta de cuantiles y metadatos entre prefijos compartidos y rutas individuales. El informe anterior aclara que su comparador del colector completo convertía las listas a FP32. Sus medidas históricas no se han cambiado ni se atribuye a aquel criterio una comprobación FP64 que no hacía.

La [mutación dirigida](mutations.json) detectó los diez cambios seleccionados. Las referencias inicial y restaurada aprobaron 24 casos CPU cada una. Los cambios reintroducen el bloqueo M0/M1, omiten huellas de política, aceptan escalas futuras, alteran la frontera exclusiva o el tramo de entrenamiento, usan la etiqueta como error, omiten el contraste con la emisión o intercambian los rasgos M3 entre activos. No es mutación global.

La [cobertura y CRAP](quality.json) usa coverage.py 7.16.1, Radon 6.0.1 y el resumidor del proyecto. Los dos módulos de producción modificados registran 717/883 sentencias y 272/392 ramas. `MechanismBinding.code` cubre 7/7 sentencias y 6/6 ramas, con CRAP 4. El constructor del enlace cubre 5/6 sentencias y 1/2 ramas, con CRAP 3,04. El constructor de `FinancialSession` cubre 71/78 sentencias y 23/30 ramas, con CRAP 28,57. Esas cifras no acreditan todas las combinaciones del coordinador ni ausencia de defectos.

## Coste de ejecución nativa

La [medición](benchmark.json) contiene ocho casos completos. Usa la RTX 4070 Laptop, PyTorch 2.14.0+cu130 y FP64. El estado oculto tiene dimensión 32, el puente tres regiones y ocho vecinos y la capacidad episódica es 16. El bloque físico máximo es de 64 filas. La cuota estimada del lector se declara en 128 MiB para esta medición, sin cambiar su valor por defecto.

Cada fase recibe cinco grupos de entradas: calentamiento y cuatro decisiones. Después se cierra. Emite `4 × activos` predicciones, admite `3 × activos` resultados maduros y censura los pendientes restantes. Los tiempos incluyen el constructor de la sesión, preparación, carga y comprobación de estados, publicación, validación y cierre. Excluyen generar los fixtures y construir los modelos. No cronometran una reapertura externa después del cierre.

Se ejecutaron dos calentamientos y tres repeticiones por alternativa, con sincronización CUDA y orden alternado. No hubo pruebas, mutaciones o consultas GitHub propias simultáneas. Había otras aplicaciones en la GPU. El proceso completo duró 789,10 s y no dejó casos sin ejecutar. El límite de inicio de nuevos casos era de 900 s, sin interrumpir una fase ya iniciada.

| Activos ficticios | Admisión | Ruta | Controlador, mediana por fase | Lector fijo equivalente, mediana por fase |
| ---: | --- | --- | ---: | ---: |
| 4 | M2 | K4cc fija | 2,31 s | 1,57 s |
| 4 | M3 | K4cc fija | 2,39 s | 1,51 s |
| 64 | M2 | K4cc fija | 9,75 s | 3,67 s |
| 64 | M3 | K4cc fija | 9,87 s | 3,45 s |
| 64 | M2 | K2c seleccionada con coeficientes dados | 7,99 s | 3,69 s |
| 64 | M3 | K2c seleccionada con coeficientes dados | 8,29 s | 3,33 s |
| 256 | M2 | K4cc fija | 36,19 s | 12,45 s |
| 256 | M3 | K4cc fija | 36,61 s | 11,71 s |

El controlador tiene sobrecoste en todos estos casos. Las predicciones comparadas coinciden exactamente en los ocho, con diferencia máxima registrada cero en FP64. Esto comprueba paridad con el lector equivalente sobre estos fixtures, no precisión predictiva sobre el mercado.

Los picos incrementales asignados por PyTorch fueron iguales entre alternativas: 685.056 bytes con cuatro activos, 10.825.216 con 64 y 12.922.880 con 256. No son toda la VRAM del sistema ni su caché reservada. El pico RAM acumulado del proceso fue de 1.601.257.472 bytes y no permite comparar RAM por alternativa. El JSON conserva todas las repeticiones y su dispersión. No se midieron energía, p95 o p99.

Los 256 identificadores reutilizan trayectorias de fixture con perturbaciones manuales. No son una muestra de 256 empresas reales ni el corpus histórico. La [trazabilidad externa](external-artifacts.json) conserva los manifiestos de archivos de los tres tamaños y las referencias de 48 ejecuciones medidas, con sus identidades y registros nativos. Los datos, pesos temporales y registros voluminosos permanecen fuera del repositorio.

Un [perfil cProfile separado](profile.json) de M3 con 64 flujos registra 2.637.201 entradas recursivas a `SessionArtifacts._payload` en el controlador frente a 258.180 en el lector fijo. También identifica trabajo de deserialización y comprobación de artefactos. Sus tiempos incluyen instrumentación e inicialización y no se mezclan con el benchmark. Orienta la siguiente revisión hacia ese recorrido. No demuestra saturación de la GPU ni que un kernel nuevo sea la solución.

No hay nuevos kernels, dependencias ni cambios C++ o CUDA. No se ejecutaron optimizadores externos, ajustes de escalas, entrenamiento, postentrenamiento, RL o evaluación científica. El selector sigue usando coeficientes dados. Su ajuste numérico, los contrastes predictivos y el registro de una campaña del modelo completo siguen pendientes. No hay mejora predictiva medida y el test final permanece cerrado.

Con el entorno uv CUDA y el enlace nativo del proyecto, los casos de admisión se ejecutan con:

```bash
uv run --no-sync pytest -q tests/memory/test_mechanism_admission.py tests/memory/test_mechanism_admission_online.py
```
