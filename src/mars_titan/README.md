# Código científico

Este directorio contiene la lógica reutilizable de MARS-TITAN. El paquete se organiza por responsabilidad y mantiene separados la preparación de datos, los modelos, las memorias, el entrenamiento, la evaluación y la simulación.

| Módulo | Responsabilidad |
| --- | --- |
| `data` | Inventariar fuentes, comprobar su disponibilidad temporal, preparar muestras multimodales con máscaras y calcular objetivos residuales |
| `models` | Referencias Ridge, boosting, recurrentes, DLinear y Transformer compacto (`baselines`), núcleo Titans-MAC y lector episódico (`titans`), adaptador de la GRU candidata nativa (`candidate`), cabeza común de cuantiles y objetivos KLPO |
| `memory` | Codec y banco episódico, escrituras M1, M2 y M3, memoria asociativa con resultados maduros, sesiones financieras y variante MARS-TITAN con ampliaciones |
| `cm` | Mecanismos C y M de CM-v1: dinámica del operador, radio numérico y medoids |
| `training` | Entrenadores cronológicos y de referencias, vistas walk-forward, objetivos del corpus, campaña con máscaras, medición de caudal y protección del aprendizaje |
| `posttraining` | Padres congelados, matriz de adaptadores y su etapa por ventana |
| `calibration` | Calibración común de cuantiles ajustada solo con el tramo de calibración |
| `evaluation` | Métricas por sesión, comparación walk-forward, estratos por presencia de modalidades y contrastes emparejados |
| `environments` | Fuentes causales de cohortes, entorno predictivo y recibos de ventana para el refuerzo |
| `simulation` | Contabilidad, reglas de mercado, cintas reconstruidas, políticas y etapa de refuerzo por ventana |
| `episodes` | Episodios cronológicos y mundos sintéticos separados del corpus real |
| `observatory` | Recolección y publicación del observatorio a partir de recibos, sin cargar modelos |
| `budget_training.py`, `reference_probe.py`, `gru_probe.py` y `profiling.py` | Sondas breves de recursos y reanudación, sin tratar sus medidas como resultados predictivos |

Estas responsabilidades siguen el [protocolo de investigación](../../docs/research/protocol.md) y la [arquitectura](../../docs/engineering/architecture.md). Los módulos rechazan entradas que incumplen sus límites o su contrato temporal, y conservan por separado los artefactos de preparación y las salidas de cada ejecución.

La evaluación no reajustará modelos a partir de la prueba final. La memoria solo recibirá la información autorizada por la política temporal. La integración con código nativo dependerá de evidencia de perfilado y de una comparación numérica con la referencia Python.
