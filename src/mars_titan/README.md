# Código científico

Este directorio contiene la lógica reutilizable de MARS-TITAN. El paquete se organiza por responsabilidad y mantiene separados la preparación de datos, los modelos de referencia y las sondas de medición.

| Módulo | Responsabilidad |
| --- | --- |
| `data` | Inventariar fuentes, comprobar su disponibilidad temporal y preparar muestras multimodales |
| `models/baselines` | Implementar las referencias Ridge, boosting y DLinear con entradas comparables |
| `budget_training.py` | Ejecutar cargas supervisadas breves y recuperables para estimar recursos |
| `reference_probe.py` | Medir las referencias tabulares sobre una muestra estricta |
| `gru_probe.py` | Medir referencias temporales y comprobar su reanudación |
| `profiling.py` | Recoger medidas locales de coste sin tratarlas como resultados predictivos |

Estas responsabilidades siguen el [protocolo de investigación](../../docs/research/protocol.md) y la [arquitectura](../../docs/engineering/architecture.md). Los módulos rechazan entradas que incumplen sus límites o su contrato temporal, y conservan por separado los artefactos de preparación y las salidas de cada ejecución.

La evaluación no reajustará modelos a partir de la prueba final. La memoria solo recibirá la información autorizada por la política temporal. La integración con código nativo dependerá de evidencia de perfilado y de una comparación numérica con la referencia Python.
