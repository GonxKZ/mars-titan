# Intervenciones sobre el puente regional

La variante opcional shared_evidence_bridge_intervention_v1 permite retirar lectores completos y cortar aristas dirigidas sin duplicar ni borrar el episodio canónico. Conserva los parámetros y el trabajo denso de sus controles. Está desactivada por defecto. El código se encuentra en c1320846f6910f14dd5efdc5062cc62ae7fc80b7 y su [especificación](../../../docs/experiments/regional-bridge-interventions-v1.md) distingue máscara, comunicación y término residual.

## Comprobaciones y límites

La selección final tiene 485 IDs distintos aprobados, de los que 125 llevan el parámetro cuda:0. No hubo fallos, errores ni omisiones en esas selecciones. Las nueve repeticiones solapadas del parser no aumentan el total. Los [recibos](verification.json) conservan selecciones, huellas e IDs. No se ejecutó la suite completa del repositorio, ningún optimizador o entrenamiento, ni se abrió el test final.

Se comprobaron paridad con el golden anterior, RNG, máscaras y cortes, JSON y factoría, gradientes, ausencia de banco, banco vacío, lecturas pobladas silenciadas, orden, bloques, trazas y presupuestos. La prueba nativa CPU/CUDA corta después de confirmar feedback y recupera la siguiente emisión, sus pesos rápidos y el evento durable, con una sola aplicación del feedback. Se reutilizó el enlace nativo existente, sin modificar código C++ o CUDA.

Las pruebas detectaron y corrigieron dos defectos antes de fijar el resultado: un buffer alterado podía convertirse en referencia de carga y una máscara alterada antes de congelar podía sellarse como válida. Los consumidores concilian ahora configuración y geometría al construirse. También rechazan un módulo intervenido oculto por metadatos sin intervención. Esa conciliación no añade copias a cada forward o prepare.

La [mutación dirigida](mutations.json) detectó diez de diez mutantes en el árbol final. Nueve pertenecen a las intervenciones regionales y uno repite la confirmación de settlement. Los controles inicial y restaurado del bloque regional aprobaron 88 casos cada uno. La tanda anterior, con 81 casos, se conserva fuera del repositorio y no se cuenta como otros nueve mutantes. El 100 % corresponde exclusivamente a esas diez sustituciones, no a todos los defectos posibles.

La [mutación de confirmación](confirmation-mutation.json) tiene además un control inicial y otro restaurado de un caso nativo cada uno. La segunda llamada se rechaza por el control de cursor. No se ha mutado el registro de idempotencia de C++.

La [calidad medida](quality.json) usa coverage.py 7.16.1 y Radon 6.0.1. En diez módulos hay 1.362 de 1.590 sentencias y 544 de 724 ramas ejecutadas. El módulo nuevo cubre 32 de 32 sentencias y 20 de 20 ramas. La validación previa al sello cubre 15 de 15 sentencias y 10 de 10 ramas, con complejidad y CRAP 14. El verificador JSON regional cubre 39 de 40 sentencias y 33 de 34 ramas, con CRAP 51,041. Las regiones no comprobadas de otros mecanismos permanecen visibles en el informe. No hay cobertura global del 100 %.

PyTorch emitió durante gradcheck CUDA un aviso de inicialización del contexto cuBLAS en el hilo de autograd. Se conserva el aviso. Las comprobaciones numéricas terminaron correctamente y no se relajaron tolerancias.

## Reconstrucciones analíticas

Los [16 paneles](analytical.json) utilizan dos episodios artificiales maduros, con valores y fechas fijados antes de ejecutar. En el fixture principal, las consultas y matrices regionales son distintas. El control redundante las iguala deliberadamente. El recibo identifica la única fuente, los parámetros, las atenciones y las vistas. Las consultas se reconstruyen desde los parámetros y el estado del fixture, fuera de las trazas de sesión. No se presentan como consultas capturadas por esas trazas.

| Condición con salida receptora fijada | Error cuadrático de reconstrucción observado |
| --- | ---: |
| Una lectura o varias vistas sin mensajes | 1,056929 |
| Comunicación solo consigo misma | 4,059446 |
| Anillo | 0 |
| Comunicación completa con vista antagonista | 2,813689 |
| Lectores redundantes | 4,059446 |
| Control directo sin mensajes, con la misma forma de salida | 0 |
| Cancelación lineal o no lineal | 1 |

El corte dirigido del mensaje hacia el receptor produce el mismo resultado negativo que el control diagonal. Retirar la tercera región conserva la reconstrucción del anillo en un paso síncrono. El registro canónico sigue presente en ambos casos. Retirar el receptor o todos los lectores da lectura cero con ese readout, sin reconstruir información borrada.

El control directo también resuelve la tarea. El éxito del anillo comprueba una contribución efectiva bajo la salida fijada y no demuestra superioridad de la arquitectura. La cancelación afecta a vistas derivadas, no borra el snapshot. Estos errores son del fixture, no MAE financiero ni evidencia de especialización aprendida.

## Coste observado

Las ejecuciones usaron una RTX 4070 Laptop con 8.188 MiB, PyTorch 2.14.0+cu130, CUDA 13, determinismo declarado y TF32/autocast desactivados. Se conservaron las aplicaciones ajenas. Cada condición tuvo tres calentamientos y diez muestras. Los percentiles son descriptivos, con pocas observaciones para estimar colas.

El [forward completo del puente](bridge-benchmark.json) tiene 40 combinaciones distintas, repetidas en dos órdenes, con B=1/256, R=3/8 y N=8/1.024. Todos los candidatos coincidieron y las salidas FP32 cumplieron atol=2e-6, rtol=1e-5 frente a la referencia CPU FP64 con los mismos valores FP32 elevados a FP64. El proceso supervisado completo duró 16,804 s.

Para B=256, R=8 y N=1.024, el p50 de J2 fue 2,247 y 3,797 ms en las dos pasadas. El control activo dio 2,431 y 3,586 ms. Todos conservaron 147.968 bytes de parámetros. La intervención añadió ocho bytes de buffer y 2.048 bytes al pico transitorio del asignador, que pasó de 7.277.056 a 7.279.104 bytes. Silenciar lectores no reduce el trabajo denso contado.

La [decisión completa](decision-benchmark.json) mide FrozenFinancialConsumer.prepare, incluidos verificación ligera, codificadores del fixture, MAC, lectura, cabeza y propuesta de estado. Utiliza FP64, R=3, K=1 y J=2. Son 20 combinaciones distintas y 40 registros en dos órdenes, con 520 llamadas. El control activo conserva exactamente todos los campos del resultado de J2 y cada repetición es determinista. La ejecución del script publicado duró 27,277 s. La primera copia privada también pasó y no aumenta esos conteos.

Para B=256 y N=1.024:

| Variante | p50 primera pasada (ms) | p50 orden inverso (ms) |
| --- | ---: | ---: |
| J2 | 11,581 | 12,208 |
| Control activo | 11,042 | 11,875 |
| Una región retirada | 11,056 | 11,476 |
| Un puente cortado | 11,193 | 10,999 |
| Todas las regiones retiradas | 22,653 | 11,210 |

La variación entre pasadas y el registro de 22,653 ms se conservan. No hay evidencia de una aceleración estable de las intervenciones. El pico transitorio de esta forma fue 25.826.816 bytes en las cinco variantes. El máximo de todo el proceso fue 104.093.696 bytes asignados y 117.440.512 reservados por Torch. El RSS máximo acumulado fue 1.519.087.616 bytes. Estos máximos incluyen el predictor, los lectores residentes y la instrumentación, sin atribución exclusiva a una variante.

La preparación y transferencia inicial del fixture, el encoding del corpus original, el colector, la persistencia nativa y el feedback quedan fuera de este tiempo. La recuperación nativa se comprueba por separado. No se midieron energía, VRAM física exclusiva ni tráfico físico de transferencias. Los bytes lógicos se publican con esa limitación.

Los scripts reproducibles son [nonlinear_bridge_cuda.py](../../../benchmarks/nonlinear_bridge_cuda.py), con --suite interventions, y [regional_decision_cuda.py](../../../benchmarks/regional_decision_cuda.py). Se ejecutan desde copias externas para conservar los resultados fuera del repositorio:

~~~bash
profile_dir=$(mktemp -d)
cp benchmarks/regional_decision_cuda.py "$profile_dir/profile_decision.py"
uv run --no-sync python "$profile_dir/profile_decision.py" --run \
  --repository "$PWD" --output "$profile_dir/run-1"
~~~

La copia ejecutada del script de decisión conserva SHA-256 e182ae45ef52c5e99b960ad34ea6d61c7303086f466b486408124a62f53611e3. Las huellas y tamaños de los originales están en [external-artifacts.json](external-artifacts.json).

No se ha medido mejora predictiva en datos financieros. Los grupos jerárquicos de #709 y las propuestas de adaptación de #711 siguen siendo trabajos separados. La implementación comprobada permanece en la PR #708 y no se promociona automáticamente al baseline.
