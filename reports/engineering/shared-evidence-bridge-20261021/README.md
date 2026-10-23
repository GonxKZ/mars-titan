# Vistas regionales sobre una evidencia común

`SharedEvidenceBridge` conserva un snapshot episódico como fuente común. Cada región calcula su atención y su transformación sobre los mismos episodios seleccionados. Los puentes comunican las vistas resultantes. El componente no almacena copias persistentes de cada episodio por región. Sí utiliza tensores derivados y copias transitorias por consulta y dispositivo.

La [especificación](../../../docs/experiments/shared-evidence-bridge-v1.md) describe ecuaciones, propiedad de los datos, presupuesto y antecedentes. El código tiene identidad `shared_evidence_bridge_v1`. No está integrado todavía en el consumidor congelado, las trazas o la configuración del predictor completo. No hay entrenamiento ni mejora predictiva medida.

## Pruebas y revisión adversarial

La [regresión seleccionada](verification.json) pasó 231 casos, con 47 marcados CUDA, sin fallos, errores ni omisiones. Incluye 52 casos nuevos del componente, diez CUDA. El XML mide 4,216 segundos y pytest informa 4,28 segundos. El único aviso procede de la prueba anterior del modo experimental de depuración de sincronizaciones de Torch. No es toda la suite del proyecto.

Las pruebas comparan las operaciones por lotes con una referencia explícita por bucles. Comprueban topologías diagonal, anillo y completa, regiones ausentes, memoria vacía, gradientes y `gradcheck`, permutación de filas, recuperación, RNG y paridad CPU/CUDA. Un caso con un solo episodio verifica que las regiones produzcan vistas diferentes del mismo ID. Otro comprueba que los puentes cambien la salida, evitando una media final que pudiera cancelar su efecto. Una sonda verifica que la proyección común reciba una sola vez las filas de evidencia, sin eje regional.

Las entradas no finitas, el presupuesto insuficiente y un snapshot de otro contexto o corte se rechazan. El checkpoint se valida antes de copiar parámetros. Se rechazan campos ausentes, formas, precisión, topología o identidad distintas. Una prueba fuerza desbordamiento antes de tanh y exige un error aunque la activación pudiera devolver un valor finito.

La [mutación dirigida](mutations.json) detectó los once cambios seleccionados. Las referencias inicial y restaurada aprobaron 42 casos CPU cada una. Los mutantes eliminan consultas o transformaciones regionales, puentes, máscaras, validación temporal, control de desbordamiento o presupuesto. También duplican el cálculo de la proyección, invierten IDs y aceptan checkpoints incompatibles. No es una puntuación sobre todas las mutaciones posibles.

La [cobertura y CRAP](quality.json) registran 153/153 sentencias y 42/42 ramas en ocho funciones del nuevo módulo. Se usaron coverage.py 7.16.1, Radon 6.0.1 y `scripts/summarize_python_quality.py`. CRAP usa cobertura de sentencias y publica ramas aparte. Los valores máximos son 20 en `forward`, 14 en la validación de configuración y 11 en la carga. La cobertura no demuestra haber comprobado cada combinación de operandos, todas las formas admitidas ni la integración completa.

El primer caso RED falló por ausencia del módulo. La primera ampliación pasó 41 casos con un aviso de conversión a escalar en la referencia. Se corrigió esa conversión mediante `detach`. También se corrigieron el orden de importaciones señalado por Ruff y una invocación de la herramienta de calidad desde un entorno vacío sin `coverage`. Las comprobaciones definitivas utilizan el entorno reproducible del proyecto.

## Coste del componente

El [protocolo](cost-protocol.json) se guardó antes de ejecutar el benchmark. Se completó una pasada de 36 condiciones y 720 muestras. Incluye 4, 64 y 256 consultas, 8, 128 y 1.024 episodios y tres u ocho regiones, con H = 32 y dos ciclos. Se midieron FP64 y FP32 explícitos, con tres calentamientos y veinte repeticiones por condición. Las tolerancias CPU/CUDA se fijaron antes: `rtol=1e-11, atol=1e-12` en FP64 y `rtol=1e-5, atol=2e-6` en FP32. Todas las condiciones conservaron IDs idénticos y cumplieron las tolerancias de valores, vistas y pesos.

Se usaron `cuda:0`, Torch 2.14.0+cu130, CUDA 13.0 y la RTX 4070 Laptop de 8.188 MiB, con driver 595.91.07 y dos hilos CPU. TF32 y autocast estaban desactivados. La GPU tenía 5.880 MiB ocupados y un 24 % de utilización antes de ejecutar. No se detuvo esa carga ajena. No hubo otras pruebas, mutaciones o consultas propias durante el cronometraje.

Con 256 consultas, 1.024 episodios y ocho regiones se observaron los siguientes valores:

| Precisión | p50 | p95 | p99 | Consultas/s | Pico asignado Torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP64 | 1,688 ms | 2,023 ms | 2,211 ms | 146.984 | 44.608.000 bytes |
| FP32 | 1,659 ms | 1,999 ms | 2,156 ms | 148.082 | 41.059.328 bytes |

El asignado anterior a esas llamadas fue de 34.981.888 y 34.539.520 bytes. El pico reservado fue de 62.914.560 y 60.817.408 bytes. Son contadores del asignador de este proceso y no la VRAM total del dispositivo. Las estimaciones de trabajo fueron 17.776.640 y 13.705.216 bytes. No se comparan esas estimaciones directamente con toda la memoria residente ni se presentan como cotas certificadas.

El [resumen](cost.json) conserva todos los tamaños, caudales y colas. Una pasada y veinte repeticiones producen medidas descriptivas. No acreditan una ventaja estable de FP32 ni una aceleración del predictor completo. El tiempo de proceso fue de 2,416 segundos internos y 7,01 externos. El pico RSS interno fue de 1.348.538.368 bytes y el externo, 1.422.108 KiB. La carga del runtime y la preparación están en el proceso, pero fuera de las latencias por llamada. No se midieron energía, transferencias físicas, registros, ocupación o contadores de caché del hardware.

Las entradas son fixtures numéricos aleatorios con semillas y fechas manuales, no un experimento financiero. La referencia CPU comprueba paridad antes de cronometrar y no se usa como comparación de velocidad. Las llamadas medidas incluyen selección, proyecciones, atención, mensajes, fusión y validaciones del componente. Excluyen codificación multimodal, recuperación del almacén nativo, MAC, cabeza, publicación y maduración de una sesión financiera.

## Reproducción y trabajo pendiente

Con el entorno uv CUDA del proyecto:

```bash
evidence_dir="$(mktemp -d)"
uv run --no-sync python reports/engineering/shared-evidence-bridge-20261021/benchmark.py --output "$evidence_dir/cost.json"
```

El benchmark exige CUDA y una salida nueva. Registra el HEAD padre `f3dee46d6331db06041615239e2cd7bb03068bac` y las huellas de las fuentes modificadas que se ejecutaron antes de su commit. Los XML, muestras, cobertura y fuentes medidas se conservan fuera de Git y se identifican en [external-artifacts.json](external-artifacts.json).

Faltan la integración tipada con el consumidor y sus trazas, los controles de lectura única y copias independientes dentro del mismo predictor, las ablaciones con igual presupuesto y la evaluación temporal real. Los nombres de región no acreditan sectores históricos, escalas aprendidas ni especialización. El bloqueo de aprendizaje y el test final siguen intactos. No hay evidencia de mejora sobre el estado del arte.
